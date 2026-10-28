"""E2Mem Core Memory Service with Evidence Anchoring, BM25+RRF, and Parent Synergy.

Provides AML-compatible Add and Search operations, storing memories in SQLite and
using Tri-Tier Evidence Grounding + C2 Channel Assembly during retrieval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .evidence_anchor import EvidenceAnchorEngine, token_count
from .model_client import ModelClient
from .pair_guard import STOPWORDS, tokenize_words


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def stable_id(prefix: str, *parts: object) -> str:
    return prefix + "_" + hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:20]


def digest_key(parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lexical(query: str, text: str, q_tokens: list[str] | None = None) -> float:
    q = q_tokens if q_tokens is not None else tokenize_words(query)
    t = tokenize_words(text)
    if not q or not t:
        return 0.0
    q_set = set(q)
    from collections import Counter
    t_counts = Counter(t)
    return sum((2.2 * n / (n + 1.2)) for w in q_set if (n := t_counts.get(w, 0)) > 0) / (len(q_set) ** 0.5)


def compute_bm25_scores(query: str, rows: list[dict]) -> list[float]:
    """Compute length-normalized, IDF-weighted BM25 scores across user memory rows."""
    q_tokens = tokenize_words(query)
    q_unique = list(dict.fromkeys(q_tokens))
    if not q_unique or not rows:
        return [0.0] * len(rows)

    n_docs = len(rows)
    doc_token_lists: list[list[str]] = []
    df: dict[str, int] = {w: 0 for w in q_unique}

    for r in rows:
        extra = ""
        if r.get("kind") == "event":
            extra = f" {r.get('aliases') or ''} {r.get('keywords') or ''} {r.get('source_time') or ''} {r.get('time_anchor') or ''}"
        text = f"{r.get('content') or ''} {r.get('evidence') or ''}{extra}"
        d_tokens = tokenize_words(text)
        doc_token_lists.append(d_tokens)
        d_set = set(d_tokens)
        for w in q_unique:
            if w in d_set:
                df[w] += 1

    avgdl = max(12.0, sum(len(t) for t in doc_token_lists) / max(1, n_docs))
    idf: dict[str, float] = {}
    for w in q_unique:
        raw_idf = math.log(1.0 + (n_docs - df[w] + 0.5) / (df[w] + 0.5))
        if w in STOPWORDS:
            raw_idf *= 0.15
        elif any(ch.isdigit() for ch in w):
            raw_idf *= 1.35
        idf[w] = raw_idf

    norm_factor = max(1.0, len(q_unique) ** 0.5)
    scores: list[float] = []
    for d_tokens in doc_token_lists:
        if not d_tokens:
            scores.append(0.0)
            continue
        dl = len(d_tokens)
        denom_norm = 1.2 * (0.35 + 0.65 * (dl / avgdl))
        s = 0.0
        for w in q_unique:
            tf = d_tokens.count(w)
            if tf > 0:
                s += idf[w] * ((2.2 * tf) / (tf + denom_norm))
        scores.append(s / norm_factor)
    return scores


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    length = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / length if length else 0.0


def compute_dense_scores(q_vec: list[float] | None, rows: list[dict]) -> list[float]:
    if not q_vec or not rows:
        return [0.0] * len(rows)
    try:
        import numpy as np
        dim = len(q_vec)
        q_arr = np.array(q_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm < 1e-9:
            return [0.0] * len(rows)
        q_unit = q_arr / q_norm

        expected_bytes_len = dim * 4
        valid_indices = []
        raw_bytes_list = []
        for idx, r in enumerate(rows):
            v = r.get("vector")
            if isinstance(v, bytes) and len(v) == expected_bytes_len:
                valid_indices.append(idx)
                raw_bytes_list.append(v)

        if not valid_indices:
            return [0.0] * len(rows)

        full_buf = b"".join(raw_bytes_list)
        mat = np.frombuffer(full_buf, dtype=np.float32).reshape(len(valid_indices), dim)
        dots = np.dot(mat, q_unit)
        norms = np.linalg.norm(mat, axis=1)
        sims = (dots / np.maximum(norms, 1e-9)).tolist()

        res = [0.0] * len(rows)
        for orig_idx, sim in zip(valid_indices, sims):
            res[orig_idx] = float(sim)
        return res
    except Exception:
        q_norm = math.sqrt(sum(a * a for a in q_vec))
        if not q_norm:
            return [0.0] * len(rows)
        res = []
        for r in rows:
            vec = decode_vector(r.get("vector"))
            if not vec or len(vec) != len(q_vec):
                res.append(0.0)
                continue
            dot = sum(a * b for a, b in zip(q_vec, vec))
            r_norm = math.sqrt(sum(b * b for b in vec))
            res.append(dot / (q_norm * r_norm) if r_norm else 0.0)
        return res


def encode_vector(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def decode_vector(stored: bytes | str | None) -> list[float]:
    if isinstance(stored, bytes):
        values = array("f")
        values.frombytes(stored)
        return values.tolist()
    return json.loads(stored or "[]")


def evidence_refs(evidence: list[str], messages: list[dict]) -> list[str]:
    found: list[str] = []
    for span in evidence:
        needle = " ".join(str(span).casefold().split())
        if len(needle) < 6:
            continue
        for msg in messages:
            ref = str(msg.get("source_ref") or "")
            haystack = " ".join(str(msg.get("content") or "").casefold().split())
            if ref and (needle in haystack or (len(haystack) >= 12 and haystack in needle)) and ref not in found:
                found.append(ref)
    return found


def pin_verbatim_evidence_and_refs(
    event_text: str,
    evidence_list: list[str],
    chunk: list[dict],
    chunk_base_idx: int,
    session_id: str,
) -> tuple[str, list[str], list[str], str, int]:
    """
    Deterministically pin an event's evidence, source_ref, keywords, and date
    to the exact verbatim message in the chunk.
    Returns: (evidence_str, refs, keywords, fact_date, matched_offset)
    """
    clean_ev_spans = [str(s).strip() for s in evidence_list if str(s).strip()]
    best_msg_idx = -1
    best_matched_span = ""
    best_score = 0.0

    # 1. First pass: exact substring match
    for offset, msg in enumerate(chunk):
        haystack = str(msg.get("content") or "")
        haystack_cf = haystack.casefold()
        for span in clean_ev_spans:
            needle_cf = span.casefold()
            if len(needle_cf) >= 6 and needle_cf in haystack_cf:
                idx = haystack_cf.find(needle_cf)
                matched_slice = haystack[idx:idx + len(span)].strip()
                score = 10.0 + len(matched_slice)
                if score > best_score:
                    best_score = score
                    best_msg_idx = offset
                    best_matched_span = matched_slice

    # 2. Second pass if no exact substring: sentence-level token overlap
    if best_msg_idx < 0:
        ev_query = f"{event_text} {' '.join(clean_ev_spans)}"
        q_tokens = set(tokenize_words(ev_query)) - STOPWORDS
        q_nums = {w for w in q_tokens if any(ch.isdigit() for ch in w)}

        for offset, msg in enumerate(chunk):
            haystack = str(msg.get("content") or "").strip()
            if not haystack:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s*", haystack) if len(s.strip()) >= 8]
            for sent in sentences:
                s_tokens = set(tokenize_words(sent)) - STOPWORDS
                s_nums = {w for w in s_tokens if any(ch.isdigit() for ch in w)}
                overlap = len(q_tokens & s_tokens)
                num_overlap = len(q_nums & s_nums)
                score = overlap + 3.0 * num_overlap
                if score > best_score and score >= 2.0:
                    best_score = score
                    best_msg_idx = offset
                    best_matched_span = sent[:260]

    # Resolve return values
    if best_msg_idx >= 0:
        matched_msg = chunk[best_msg_idx]
        ref = str(matched_msg.get("source_ref") or f"{session_id}:{chunk_base_idx + best_msg_idx}")
        refs = [ref]
        msg_date = str(matched_msg.get("timestamp") or "")
        evidence_str = best_matched_span or ("; ".join(clean_ev_spans[:2]) if clean_ev_spans else event_text[:200])
    else:
        # Do not fabricate a pseudo-attribution to chunk[0]
        refs = []
        msg_date = ""
        evidence_str = "; ".join(clean_ev_spans[:2]) if clean_ev_spans else event_text[:200]

    # Compute high-quality keywords
    kw_words = [w for w in tokenize_words(f"{event_text} {evidence_str}") if len(w) > 2 and w not in STOPWORDS]
    sorted_kws = sorted(
        dict.fromkeys(kw_words),
        key=lambda w: (0 if any(c.isdigit() for c in w) else 1, -len(w)),
    )[:10]

    return evidence_str, refs, sorted_kws, msg_date, best_msg_idx


def extract_uncovered_user_factlets(
    chunk: list[dict],
    extracted_events: list[dict],
    session_id: str,
    chunk_base_idx: int,
    max_per_chunk: int = 2,
) -> list[dict]:
    """Deterministically extract verbatim factlet events from user sentences omitted by the LLM extractor."""
    ev_text_union = " ".join(str(ev.get("event_text") or "") for ev in extracted_events)
    ev_words = {w for w in tokenize_words(ev_text_union) if len(w) > 2 and w not in STOPWORDS}
    ev_nums = {
        w for w in tokenize_words(ev_text_union)
        if any(ch.isdigit() for ch in w) and not (len(w) == 4 and w.startswith(("19", "20")))
    }

    fallback_date = ""
    for ev in extracted_events:
        d_str = str(ev.get("fact_date_resolved") or "").strip()
        if d_str:
            fallback_date = d_str
            break

    candidates: list[tuple[float, dict]] = []
    for offset, msg in enumerate(chunk):
        if msg.get("role") != "user":
            continue
        raw_content = str(msg.get("content") or "").strip()
        if not raw_content or len(raw_content) >= 2500:
            continue

        clean_text = re.sub(r"\s*->->\s*\d+,\d+\s*$", "", raw_content).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", clean_text) if s.strip()]

        for s_idx, sent in enumerate(sentences):
            sent_clean = sent
            if sent_clean.endswith("?"):
                m_tail = re.split(
                    r",\s*(?:can you|could you|do you|what do you|how can i|should i)\b",
                    sent_clean,
                    flags=re.IGNORECASE,
                )
                if len(m_tail) > 1 and len(m_tail[0].strip()) >= 20:
                    sent_clean = m_tail[0].strip() + "."
                else:
                    continue

            s_words = {w for w in tokenize_words(sent_clean) if len(w) > 2 and w not in STOPWORDS}
            if len(s_words) < 4:
                continue

            s_nums = {
                w for w in tokenize_words(sent_clean)
                if any(ch.isdigit() for ch in w) and not (len(w) == 4 and w.startswith(("19", "20")))
            }
            overlap = len(s_words & ev_words) / max(1, len(s_words))
            missing_nums = s_nums - ev_nums
            has_action = bool(
                re.search(
                    r"\b(allocated|completed|finished|bought|adopted|acquired|tried|learned|started|considering|brew|brewing|personal best|scale)\b",
                    sent_clean.casefold(),
                )
            )

            if overlap >= 0.30 and not missing_nums:
                continue
            if not missing_nums and not has_action:
                continue

            score = (
                3.5 * len(missing_nums)
                + (2.5 if has_action else 0.0)
                + 1.5 * (1.0 - overlap)
                + (1.0 if re.search(r"[$%]|\b(minutes|seconds|hours|books|scale|miles|km)\b", sent_clean.casefold()) else 0.0)
            )
            if score < 3.2:
                continue

            ref = str(msg.get("source_ref") or f"{session_id}:{chunk_base_idx + offset}")
            msg_time = str(msg.get("timestamp") or fallback_date)
            kw_list = sorted(
                s_words,
                key=lambda w: (0 if any(c.isdigit() for c in w) else 1, -len(w)),
            )[:10]

            candidates.append((
                score,
                {
                    "msg_offset": offset,
                    "sent_idx": s_idx,
                    "turn_offset": offset * 10 + s_idx,
                    "content": sent_clean[:240],
                    "evidence": clean_text[:260],
                    "source_time": msg_time,
                    "source_refs": [ref],
                    "keywords": kw_list,
                    "_words": s_words,
                },
            ))

    candidates.sort(key=lambda x: -x[0])
    factlets: list[dict] = []
    for _, cand in candidates:
        if len(factlets) >= max_per_chunk:
            break
        c_words = cand.pop("_words")
        if len(c_words & ev_words) / max(1, len(c_words)) >= 0.45:
            continue
        factlets.append(cand)
        ev_words.update(c_words)

    return factlets


def segment_document_clauses(text: str) -> list[str]:
    """Deterministically slice a long document or structured text into coherent sections/clauses.
    Identifies natural structural boundaries:
    - Bracketed tags (e.g. [Turn 1 ...], [Artifact 1 ...], [Section ...])
    - Unbracketed headers (e.g. Artifact 1 —, Section 1., Step 1:)
    - Markdown headers (e.g. #, ##, ###)
    - Double-newline paragraph boundaries for long continuous text blocks.
    """
    if len(text) < 1200:
        return [text]

    pattern = r"(?=(?:\[(?:Turn|Artifact|Section|Chapter|Step|Rule)\s+\d+[^\]]*\]|(?:Artifact|Section|Chapter|Step|Rule)\s+\d+\s*[-—:]|\n#{1,4}\s+|\n\s*(?:Step|Rule)\s+\d+[:.]))"
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]

    if len(parts) <= 1:
        # Fallback to paragraph-based chunking (~600-1200 chars)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        cur = []
        cur_len = 0
        for p in paragraphs:
            if cur_len + len(p) > 1000 and cur:
                chunks.append("\n\n".join(cur))
                cur = [p]
                cur_len = len(p)
            else:
                cur.append(p)
                cur_len += len(p)
        if cur:
            chunks.append("\n\n".join(cur))
        return chunks if chunks else [text]

    return parts


class MemoryService:
    def __init__(
        self,
        db_path: str = ":memory:",
        mode: str = "simulator",
        read_only: bool = False,
        enable_jev: bool = False,
        trace_file: str | Path | None = None,
    ):
        self.mode = mode
        self.db_path = db_path
        self.read_only = read_only
        self.enable_jev = enable_jev
        self.lock = threading.RLock()
        self.trace_lock = threading.Lock()

        if trace_file is not None:
            self.trace_path: Path | None = Path(trace_file)
        elif os.environ.get("E2MEM_ADD_TRACE_FILE"):
            self.trace_path = Path(os.environ["E2MEM_ADD_TRACE_FILE"])
        elif db_path != ":memory:":
            self.trace_path = Path(db_path).parent / "add_traces.jsonl"
        else:
            self.trace_path = None
        if db_path != ":memory:":
            try:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        if read_only and db_path != ":memory:":
            self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        else:
            self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if not read_only:
            self._init_db()

        self.cache_db: sqlite3.Connection | None = None
        if db_path != ":memory:":
            cache_path = Path(str(db_path) + ".cache.sqlite3")
            if cache_path.exists():
                try:
                    self.cache_db = sqlite3.connect(
                        f"file:{cache_path}?mode=ro", uri=True, check_same_thread=False
                    )
                    self.cache_db.row_factory = sqlite3.Row
                except Exception:
                    self.cache_db = None

        self.model = ModelClient(mode)
        self.anchor_engine = EvidenceAnchorEngine(self.db, enable_jev=enable_jev)

    def _init_db(self) -> None:
        with self.lock:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS requests(
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    body_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories(
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    parent_id TEXT,
                    turn_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    source_time TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    source_refs TEXT NOT NULL,
                    aliases TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    time_anchor TEXT NOT NULL,
                    topic_labels TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mem_user_kind ON memories(user_id, kind);
                CREATE INDEX IF NOT EXISTS idx_mem_parent ON memories(parent_id);
            """)
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            if self.cache_db is not None:
                try:
                    self.cache_db.close()
                except Exception:
                    pass
            self.db.close()

    def _get_query_vector(self, query: str) -> list[float]:
        if not hasattr(self, "_query_vector_cache"):
            self._query_vector_cache: dict[str, list[float]] = {}
        if query in self._query_vector_cache:
            return self._query_vector_cache[query]

        if self.cache_db is not None:
            try:
                key = digest_key([self.model.embedding_model, query])
                row = self.cache_db.execute(
                    "SELECT vector FROM vector_cache WHERE key=? LIMIT 1", (key,)
                ).fetchone()
                if row and row["vector"]:
                    vec = decode_vector(row["vector"])
                    self._query_vector_cache[query] = vec
                    return vec
            except Exception:
                pass
        query_vectors, _ = self.model.embed([query])
        vec = query_vectors[0] if query_vectors else []
        if vec:
            self._query_vector_cache[query] = vec
            if self.cache_db is not None:
                try:
                    key = digest_key([self.model.embedding_model, query])
                    self.cache_db.execute(
                        "INSERT OR IGNORE INTO vector_cache VALUES(?,?,?)",
                        (key, encode_vector(vec), "query"),
                    )
                    self.cache_db.commit()
                except Exception:
                    pass
        return vec

    def _validate_add(self, payload: dict) -> None:
        for key in ("request_id", "user_id", "session_id"):
            if not isinstance(payload.get(key), str) or not payload[key]:
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise ValueError("messages must be a nonempty array")
        for msg in payload["messages"]:
            if not isinstance(msg, dict) or not isinstance(msg.get("role"), str) or not isinstance(msg.get("content"), str) or not msg["content"]:
                raise ValueError("each message needs role and nonempty textual content")

    def add(self, payload: dict) -> tuple[dict, dict]:
        self._validate_add(payload)
        start = time.perf_counter()
        request_id, user_id, session_id = (payload[k] for k in ("request_id", "user_id", "session_id"))
        body_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result = {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

        with self.lock:
            existing = self.db.execute("SELECT body_hash FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if existing:
                if existing["body_hash"] != body_hash:
                    raise RuntimeError("request_id reused with different content")
                return result, {"duplicate": True}

        messages = payload["messages"]
        has_long_doc = any(m.get("role") == "user" and len(str(m.get("content") or "")) > 1200 for m in messages)

        # 1. Segment into logical units for extraction
        segmented_units: list[dict] = []
        for msg_idx, msg in enumerate(messages):
            content = str(msg.get("content") or "")
            ref = str(msg.get("source_ref") or f"{session_id}:{msg_idx}")
            t_stamp = str(msg.get("timestamp") or "")
            role = msg.get("role", "user")

            if role == "user" and len(content) > 1200:
                clauses = segment_document_clauses(content)
                for c_idx, clause in enumerate(clauses):
                    segmented_units.append({
                        "parent_msg_idx": msg_idx,
                        "clause_idx": c_idx,
                        "role": role,
                        "content": clause,
                        "source_ref": f"{ref}#c{c_idx}" if len(clauses) > 1 else ref,
                        "timestamp": t_stamp,
                    })
            else:
                extract_content = content
                if role == "assistant" and len(content) > 2500:
                    extract_content = content[:2500] + "\n...[assistant detailed elaboration omitted for memory extraction]"
                segmented_units.append({
                    "parent_msg_idx": msg_idx,
                    "clause_idx": 0,
                    "role": role,
                    "content": extract_content,
                    "source_ref": ref,
                    "timestamp": t_stamp,
                })

        # Batch units into balanced extraction chunks (~1000-2200 tokens / 4-8 units)
        chunks: list[list[dict]] = []
        cur_chunk: list[dict] = []
        cur_tokens = 0
        max_units = 6 if has_long_doc else 8
        max_tokens = 2200 if has_long_doc else 2500

        for unit in segmented_units:
            u_tokens = estimate_tokens(unit["content"])
            if cur_chunk and (len(cur_chunk) >= max_units or cur_tokens + u_tokens > max_tokens):
                chunks.append(cur_chunk)
                cur_chunk = [unit]
                cur_tokens = u_tokens
            else:
                cur_chunk.append(unit)
                cur_tokens += u_tokens
        if cur_chunk:
            chunks.append(cur_chunk)

        # Parallel chunk extraction for high throughput on multi-chunk documents
        if len(chunks) <= 1:
            record, _ = self.model.extract(chunks[0], session_id, user_id, prior_context=[]) if chunks else ({}, {})
            extracted = [record] if chunks else []
        else:
            def _extract_chunk(idx: int) -> tuple[int, dict]:
                prior = chunks[idx - 1][-2:] if idx > 0 else []
                rec, _ = self.model.extract(chunks[idx], session_id, user_id, prior_context=prior)
                return idx, rec

            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as executor:
                chunk_results = list(executor.map(_extract_chunk, range(len(chunks))))
                chunk_results.sort(key=lambda x: x[0])
                extracted = [r[1] for r in chunk_results]

        now = datetime.now(timezone.utc).isoformat()
        records = []

        # 1. Store RAW messages (and child clauses if long document segmented)
        raw_records_by_idx: dict[int, dict] = {}
        for index, msg in enumerate(messages):
            ref = str(msg.get("source_ref") or f"{session_id}:{index}")
            r_rec = {
                "id": stable_id("raw", request_id, index),
                "kind": "raw",
                "parent_id": None,
                "turn_index": index,
                "role": msg["role"],
                "source_time": str(msg.get("timestamp") or ""),
                "content": msg["content"],
                "evidence": "",
                "source_refs": [ref],
                "aliases": [],
                "keywords": [],
                "event_type": "",
                "time_anchor": "",
                "topic_labels": [],
            }
            records.append(r_rec)
            raw_records_by_idx[index] = r_rec

        if has_long_doc:
            for unit in segmented_units:
                p_idx = unit["parent_msg_idx"]
                c_idx = unit["clause_idx"]
                if c_idx > 0 or any(u["parent_msg_idx"] == p_idx and u["clause_idx"] > 0 for u in segmented_units):
                    parent_raw_id = stable_id("raw", request_id, p_idx)
                    clause_kws = [w for w in tokenize_words(unit["content"]) if len(w) > 2 and w not in STOPWORDS][:8]
                    records.append({
                        "id": stable_id("raw_c", request_id, p_idx, c_idx),
                        "kind": "raw",
                        "parent_id": parent_raw_id,
                        "turn_index": p_idx * 100 + c_idx,
                        "role": unit["role"],
                        "source_time": unit["timestamp"],
                        "content": unit["content"],
                        "evidence": "",
                        "source_refs": [unit["source_ref"]],
                        "aliases": [],
                        "keywords": clause_kws,
                        "event_type": "clause",
                        "time_anchor": "",
                        "topic_labels": [],
                    })

        # 2. Store EPISODES and EVENTS (Extraction v3.0 Dynamic Capacity & Entity Grounding)
        for index, item in enumerate(extracted):
            episode_id = stable_id("ep", request_id, index)
            chunk = chunks[index]
            base_msg_idx = chunk[0].get("parent_msg_idx", index * 8) if chunk else index * 8
            chunk_time = next((str(m.get("timestamp") or "") for m in chunk if m.get("timestamp")), "")
            summary = str(item.get("summary") or item.get("episode") or " ".join(m["content"] for m in chunk)).strip()
            if summary:
                records.append({
                    "id": episode_id,
                    "kind": "episode",
                    "parent_id": None,
                    "turn_index": base_msg_idx,
                    "role": "",
                    "source_time": chunk_time,
                    "content": summary,
                    "evidence": str(item.get("episode") or "")[:1000],
                    "source_refs": [str(m.get("source_ref") or "") for m in chunk if m.get("source_ref")],
                    "aliases": [],
                    "keywords": [],
                    "event_type": "",
                    "time_anchor": "",
                    "topic_labels": [],
                })

            llm_events = (item.get("episode_events") or [])[:16]
            for event_index, event in enumerate(llm_events):
                value = str(event.get("event_text") or "").strip()
                if value:
                    evidence_list = [str(x) for x in (event.get("evidence") or [])[:3]]
                    pin_ev, pin_refs, pin_kws, pin_date, matched_off = pin_verbatim_evidence_and_refs(
                        event_text=value,
                        evidence_list=evidence_list,
                        chunk=chunk,
                        chunk_base_idx=base_msg_idx,
                        session_id=session_id,
                    )
                    event_date = str(event.get("fact_date_resolved") or "").strip() or pin_date or chunk_time
                    llm_kws = [str(x) for x in (event.get("keywords") or [])[:12] if str(x).strip()]

                    # Boundary / Abstention annotation
                    if event.get("unspecified_details") and "[specific" not in value.lower() and "[unspecified" not in value.lower():
                        value = f"{value} [Note: Specific details were not disclosed in source]"

                    records.append({
                        "id": stable_id("ev", request_id, index, event_index),
                        "kind": "event",
                        "parent_id": episode_id,
                        "turn_index": base_msg_idx,
                        "role": "",
                        "source_time": event_date,
                        "content": value,
                        "evidence": pin_ev,
                        "source_refs": pin_refs,
                        "aliases": [str(x) for x in (event.get("aliases") or [])[:8]],
                        "keywords": llm_kws if llm_kws else pin_kws,
                        "event_type": str(event.get("event_type") or ""),
                        "time_anchor": str(event.get("time_anchor_text") or ""),
                        "topic_labels": [],
                    })
                    if matched_off >= 0:
                        matched_unit = chunk[matched_off]
                        p_msg_idx = matched_unit.get("parent_msg_idx", base_msg_idx + matched_off)
                        raw_rec = raw_records_by_idx.get(p_msg_idx)
                        if raw_rec and not raw_rec["evidence"]:
                            raw_rec["evidence"] = pin_ev
                            raw_rec["keywords"] = pin_kws

            # Supplementary factlets for dialogue chunks
            if not has_long_doc:
                supplementary_factlets = extract_uncovered_user_factlets(
                    chunk=chunk,
                    extracted_events=llm_events,
                    session_id=session_id,
                    chunk_base_idx=base_msg_idx,
                )
                if supplementary_factlets and summary:
                    ep_fact_str = " | ".join(f["content"] for f in supplementary_factlets)
                    for r in reversed(records):
                        if r["id"] == episode_id:
                            r["evidence"] = ep_fact_str[:600]
                            break
                for f_item in supplementary_factlets:
                    msg_idx = base_msg_idx + f_item.get("msg_offset", f_item.get("turn_offset", 0) // 10)
                    raw_rec = raw_records_by_idx.get(msg_idx)
                    if raw_rec is not None and not raw_rec["evidence"]:
                        raw_rec["evidence"] = f_item["content"]
                        raw_rec["keywords"] = f_item["keywords"]

        embed_inputs = []
        for r in records:
            if r["kind"] == "event" and (r["time_anchor"] or r["source_time"] or r["keywords"]):
                kw_str = " ".join(r["keywords"][:6])
                embed_inputs.append(
                    f"{r['content']} | {r['time_anchor']} {r['source_time']} | {kw_str}".strip(" |")
                )
            elif r["kind"] in ("raw", "episode") and r["evidence"] and r["evidence"][:40] not in r["content"]:
                embed_inputs.append(f"{r['evidence']} | {r['content']}")
            else:
                embed_inputs.append(r["content"])
        vectors, _ = self.model.embed(embed_inputs)

        with self.lock:
            existing = self.db.execute("SELECT body_hash FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if existing:
                return result, {"duplicate": True}

            self.db.execute("BEGIN")
            try:
                self.db.execute("INSERT INTO requests VALUES(?,?,?,?)", (request_id, user_id, session_id, body_hash))
                for record, vector in zip(records, vectors):
                    self.db.execute("""INSERT INTO memories
                        (id,user_id,session_id,request_id,kind,parent_id,turn_index,role,source_time,
                         content,evidence,vector,created_at,source_refs,aliases,keywords,event_type,time_anchor,topic_labels)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (record["id"], user_id, session_id, request_id, record["kind"], record["parent_id"],
                         record["turn_index"], record["role"], record["source_time"], record["content"],
                         record["evidence"], encode_vector(vector), now,
                         json.dumps(record["source_refs"]), json.dumps(record["aliases"]),
                         json.dumps(record["keywords"]), record["event_type"], record["time_anchor"],
                         json.dumps(record["topic_labels"])))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

        raw_count = len([r for r in records if r["kind"] == "raw"])
        ep_count = len([r for r in records if r["kind"] == "episode"])
        ev_count = len([r for r in records if r["kind"] == "event"])
        stats = {
            "messages": len(messages),
            "raw": raw_count,
            "episodes": ep_count,
            "events": ev_count,
            "extracted": extracted,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }

        # Persist full Add request & extraction trace
        if self.trace_path is not None:
            trace_entry = {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": now,
                "messages_count": len(messages),
                "messages": messages,
                "extracted_chunks": extracted,
                "stored_summary": {
                    "raw": raw_count,
                    "episodes": ep_count,
                    "events": ev_count,
                    "total_records": len(records),
                },
                "latency_ms": stats["latency_ms"],
            }
            try:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trace_lock:
                    with open(self.trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        return result, stats

    def search(
        self,
        payload: dict,
        budget: int = 8000,
        enable_pair_guard: bool = True,
        allowed_request_ids: set[str] | None = None,
        use_rrf: bool = True,
        channel_quotas: dict[str, float] | None = None,
    ) -> tuple[dict, dict]:
        query = payload.get("query")
        user_id = payload.get("user_id")
        top_k = payload.get("top_k", 100)

        if not isinstance(query, str) or not query or not isinstance(user_id, str) or not user_id:
            raise ValueError("query and user_id must be nonempty strings")
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise ValueError("top_k must be an integer between 1 and 100")

        start = time.perf_counter()
        with self.lock:
            rows = [dict(x) for x in self.db.execute("SELECT * FROM memories WHERE user_id=? ORDER BY rowid ASC", (user_id,))]

        if allowed_request_ids is not None:
            rows = [r for r in rows if r.get("request_id") in allowed_request_ids]

        if not rows:
            return {"data": []}, {"returned": 0, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

        # Chronological sort key: respects session timeline and source message index (P1-E fix)
        def _get_sort_key(r: dict) -> tuple:
            st = str(r.get("source_time") or "")
            src_idx = None
            s_refs = r.get("source_refs")
            if s_refs:
                if isinstance(s_refs, str):
                    try:
                        parsed = json.loads(s_refs)
                        if isinstance(parsed, list) and parsed:
                            s_refs = parsed
                    except Exception:
                        pass
                if isinstance(s_refs, list) and s_refs:
                    first_ref = str(s_refs[0])
                    if ":" in first_ref:
                        try:
                            src_idx = int(first_ref.split(":")[-1])
                        except Exception:
                            pass
                    elif first_ref.isdigit():
                        src_idx = int(first_ref)

            sess = str(r.get("session_id") or "")
            sess_num = 0
            m = re.search(r"(\d+)", sess)
            if m:
                sess_num = int(m.group(1))

            t_idx = int(r.get("turn_index") or 0)
            order_idx = src_idx if src_idx is not None else t_idx
            return (st, sess_num, order_idx)

        rows.sort(key=_get_sort_key)

        for idx, row in enumerate(rows, 1):
            row["seq_id"] = idx

        raw_all = [r for r in rows if r["kind"] == "raw"]
        raw_parents = [r for r in raw_all if not r.get("parent_id")]
        raw_for_metrics = raw_parents if raw_parents else raw_all
        avg_raw_len = (sum(len(r["content"]) for r in raw_for_metrics) / len(raw_for_metrics)) if raw_for_metrics else 0.0
        undated_raw_ratio = (
            sum(1 for r in raw_all if not r.get("source_time")) / len(raw_all)
        ) if raw_all else 0.0
        has_role_prefix = any(
            str(r.get("content") or "").startswith(("User:", "Assistant:", "Current user persona:"))
            for r in raw_all[:10]
        )
        undated_dialogue_mode = (avg_raw_len < 2500) and (undated_raw_ratio >= 0.80) and (not has_role_prefix)

        # Score rows with dense + classic lexical + length-normalized IDF BM25
        q_vec = self._get_query_vector(query)
        q_tokens_lex = tokenize_words(query)
        bm25_list = compute_bm25_scores(query, rows)
        dense_list = compute_dense_scores(q_vec, rows)

        for row, b_score, d_score in zip(rows, bm25_list, dense_list):
            row["lex"] = lexical(query, row["content"] + " " + (row["evidence"] or ""), q_tokens=q_tokens_lex)
            row["bm25"] = b_score
            row["dense"] = d_score

        # Multi-channel scoring with RRF rank protection, Event-first prior, and Preference Alignment
        channels = {}
        query_words = set(tokenize_words(query))
        advisory_pattern = re.compile(
            r"\b(how (would|should|can|do) (you|i)|suggest|recommend|advice|advise|approach|decide|decision|guidance|plan)\b",
            re.IGNORECASE,
        )
        preference_pattern = re.compile(
            r"\b(i prefer|my preference|i prioritize|i favor|i usually choose|i like to|i always want|i tend to prefer)\b",
            re.IGNORECASE,
        )
        is_advisory = bool(advisory_pattern.search(query))

        for kind in ("event", "episode", "raw"):
            channel_rows = [r for r in rows if r["kind"] == kind]
            if not channel_rows:
                channels[kind] = []
                continue
            max_lex = max((r["lex"] for r in channel_rows), default=1.0) or 1.0
            max_bm25 = max((r["bm25"] for r in channel_rows), default=1.0) or 1.0

            if use_rrf:
                by_dense = sorted(channel_rows, key=lambda r: (-r["dense"], r["id"]))
                by_bm25 = sorted(channel_rows, key=lambda r: (-r["bm25"], r["id"]))
                dense_rank = {r["id"]: idx for idx, r in enumerate(by_dense, 1)}
                bm25_rank = {r["id"]: idx for idx, r in enumerate(by_bm25, 1)}
            else:
                dense_rank = {}
                bm25_rank = {}

            for r in channel_rows:
                if use_rrf:
                    rrf_bonus = (
                        0.5 * (60.0 / (60.0 + dense_rank[r["id"]]))
                        + 0.5 * (60.0 / (60.0 + bm25_rank[r["id"]]))
                    )
                    r["score"] = (
                        0.52 * r["dense"]
                        + 0.24 * (r["lex"] / max_lex)
                        + 0.24 * (r["bm25"] / max_bm25)
                        + 0.08 * rrf_bonus
                    )
                else:
                    r["score"] = 0.55 * r["dense"] + 0.45 * (r["lex"] / max_lex)

                sentiment_chit_chat = bool(re.search(
                    r"\b(agrees with|expresses gratitude|expresses appreciation|thank(s|ed)|feel(s|ing)? happy|shared feelings?|said hello|goodbye|nice to chat)\b",
                    r.get("content") or "",
                    re.IGNORECASE,
                ))
                if sentiment_chit_chat and kind == "event":
                    r["score"] -= 0.05
                else:
                    r["score"] += {"event": 0.08, "episode": 0.03, "raw": 0.0}[kind]

                if undated_dialogue_mode and kind == "raw" and r.get("role") == "user":
                    r["score"] += 0.035
                if kind == "raw" and r.get("dense", 0.0) >= 0.44:
                    r["score"] += 0.04
                if is_advisory and preference_pattern.search(r.get("content") or ""):
                    r["score"] += 0.12
                if kind == "event" and (r["source_time"] or r["time_anchor"]):
                    d_tokens = set(tokenize_words((r["source_time"] or "") + " " + (r["time_anchor"] or "")))
                    if d_tokens & query_words:
                        r["score"] += 0.05

            channels[kind] = sorted(channel_rows, key=lambda r: (-r["score"], r["id"]))

        # Adaptive Long-Artifact Mode (for long specification/rulebook corpora such as CLBench)
        max_raw_len = max((len(r.get("content") or "") for r in raw_all), default=0)
        is_long_artifact = bool(raw_all and (avg_raw_len >= 2400 or (max_raw_len >= 6000 and avg_raw_len >= 1500 and len(raw_all) <= 15)))
        if is_long_artifact:
            raw_chron = sorted(raw_all, key=lambda r: r["seq_id"])
            effective_budget = max(budget, 24000)
            picked_ids = set()
            used_tok = 0
            # Always preserve first document chunk (core definitions/rules) if it fits budget
            first_cost = token_count(raw_chron[0]["content"])
            if first_cost <= effective_budget:
                picked_ids.add(raw_chron[0]["id"])
                used_tok += first_cost

            for r in channels["raw"]:
                if r["id"] in picked_ids:
                    continue
                t_cost = token_count(r["content"])
                if used_tok + t_cost <= effective_budget:
                    picked_ids.add(r["id"])
                    used_tok += t_cost
            selected_raw = [r for r in raw_chron if r["id"] in picked_ids]

            output = [
                {"id": r["id"], "content": r["content"], "score": round(float(r.get("score") or 1.0), 6)}
                for r in selected_raw
            ][:top_k]
            stats = {
                "returned": len(output),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
            return {"data": output}, stats

        # Parent-Episode Multi-Hit Synergy (Bottom-Up and Top-Down)
        event_hits = channels.get("event", [])[:30]
        grouped: dict[str, list[dict]] = {}
        for event in event_hits:
            p_id = event.get("parent_id")
            if p_id:
                grouped.setdefault(p_id, []).append(event)

        parent_by_id = {r["id"]: r for r in channels.get("episode", [])}
        synergy_events: list[dict] = []
        seen_event_ids: set[str] = set()

        # 1. Bottom-Up: cluster matching events under same parent episode
        for parent_id, matches in grouped.items():
            parent = parent_by_id.get(parent_id)
            if not parent:
                continue
            matches.sort(key=lambda r: (-r["score"], r["id"]))
            best = matches[0]["score"]
            second = matches[1]["score"] if len(matches) > 1 else 0.0
            group_bonus = min(0.28, 0.05 * (len(matches) - 1))
            group_score = best + 0.20 * second + group_bonus
            for rank_in_ep, ev in enumerate(matches[:3]):
                ev_copy = dict(ev)
                if rank_in_ep == 0:
                    ev_copy["score"] = group_score + ev["score"] * 0.01
                elif rank_in_ep == 1:
                    ev_copy["score"] = 0.65 * group_score + 0.35 * ev["score"]
                else:
                    ev_copy["score"] = ev["score"] + 0.03
                ev_copy["parent_content"] = parent["content"]
                synergy_events.append(ev_copy)
                seen_event_ids.add(ev["id"])

        # 2. Top-Down: top-scoring episodes pull in their child events into synergy pool
        child_events_by_parent: dict[str, list[dict]] = {}
        for r in rows:
            if r["kind"] == "event" and r.get("parent_id"):
                child_events_by_parent.setdefault(r["parent_id"], []).append(r)

        for top_ep in channels.get("episode", [])[:6]:
            ep_score = float(top_ep.get("score") or 0.0)
            if ep_score >= 0.40:
                child_list = child_events_by_parent.get(top_ep["id"], [])
                for ch_ev in child_list:
                    if ch_ev["id"] not in seen_event_ids:
                        ch_copy = dict(ch_ev)
                        ch_copy["score"] = 0.75 * ep_score + 0.25 * float(ch_ev.get("score") or 0.0)
                        ch_copy["parent_content"] = top_ep["content"]
                        synergy_events.append(ch_copy)
                        seen_event_ids.add(ch_ev["id"])

        synergy_events.sort(key=lambda r: (-r["score"], r["id"]))

        # Expand pool (EV64 + EP32 + RAW96) + Top-16 Dense Protection per channel
        for ev in channels.get("event", [])[:64]:
            if ev["id"] not in seen_event_ids:
                ev_copy = dict(ev)
                parent = parent_by_id.get(ev.get("parent_id"))
                if parent:
                    ev_copy["parent_content"] = parent["content"]
                synergy_events.append(ev_copy)
                seen_event_ids.add(ev["id"])

        episode_hits = [dict(r) for r in channels.get("episode", [])[:32]]
        raw_hits = [dict(r) for r in channels.get("raw", [])[:96]]

        # Ensure top-16 pure dense RAW hits are never expelled from candidate pool
        seen_raw_ids = {r["id"] for r in raw_hits}
        for r in sorted(channels.get("raw", []), key=lambda x: (-x["dense"], x["id"]))[:16]:
            if r["id"] not in seen_raw_ids:
                raw_hits.append(dict(r))
                seen_raw_ids.add(r["id"])

        combined_candidates = synergy_events[:64] + episode_hits + raw_hits
        combined_candidates.sort(key=lambda r: (-r["score"], r["id"]))

        # Elastic Budget Routing based on query & physical modality
        effective_budget = budget
        if budget >= 8000:
            if is_advisory or any(opt in query for opt in [" A.", " B.", " C.", " D.", "Option ", "preference"]):
                effective_budget = min(budget, 7500)

        # Build in-memory raw_cache to eliminate all SQLite queries during context assembly
        raw_cache = {
            "by_id": {},
            "by_ref": {},
            "by_req": {},
            "episodes": {},
        }
        for r in rows:
            r_kind = r.get("kind")
            r_id = str(r.get("id") or "")
            if r_kind == "episode" and r_id:
                raw_cache["episodes"][r_id] = r
            elif r_kind == "raw":
                if r_id:
                    raw_cache["by_id"][r_id] = r
                req = r.get("request_id")
                if req:
                    raw_cache["by_req"].setdefault(req, []).append(r)
                s_refs = r.get("source_refs")
                if s_refs:
                    if isinstance(s_refs, str):
                        try:
                            parsed_refs = json.loads(s_refs)
                            if isinstance(parsed_refs, list):
                                for ref in parsed_refs:
                                    if ref:
                                        raw_cache["by_ref"][str(ref)] = r
                            elif isinstance(parsed_refs, str) and parsed_refs:
                                raw_cache["by_ref"][parsed_refs] = r
                        except Exception:
                            for ref in s_refs.split(","):
                                ref = ref.strip().strip('"').strip("'")
                                if ref:
                                    raw_cache["by_ref"][ref] = r
                    elif isinstance(s_refs, list):
                        for ref in s_refs:
                            if ref:
                                raw_cache["by_ref"][str(ref)] = r

        # Use EvidenceAnchorEngine to assemble high-fidelity anchored context
        assembled_data = self.anchor_engine.assemble_anchored_context(
            retrieval_hits=combined_candidates,
            user_id=user_id,
            budget=effective_budget,
            enable_pair_guard=enable_pair_guard,
            query=query,
            channel_quotas=channel_quotas,
            undated_dialogue_mode=undated_dialogue_mode,
            raw_cache=raw_cache,
        )

        output = assembled_data[:top_k]
        # Atomic Pair cohesion protection: prevent top_k cutoff from slicing a paired item.
        # Strict contract constraint: len(output) must NEVER exceed top_k. If a pair is severed
        # at the top_k boundary, drop the isolated half-pair to preserve atomicity without exceeding top_k.
        if top_k < len(assembled_data) and output:
            last_item = output[-1]
            last_partner = last_item.get("partner_id")
            if last_partner and len(assembled_data) > top_k:
                next_item = assembled_data[top_k]
                if next_item.get("id") == last_partner:
                    output.pop()
        stats = {
            "returned": len(output),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
        return {"data": output}, stats
