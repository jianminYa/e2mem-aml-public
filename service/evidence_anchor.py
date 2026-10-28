"""Evidence Anchor and Dual-Representation Assembly Engine for E2Mem.

Separates retrieval index representation (fine-grained Events) from answer evidence
delivery (high-fidelity Raw Source Spans), while actively preserving paired
counter-evidence and temporal progressions.

Strictly compliant with AML competition rules: NO prompt rewriting, NO answer hints,
and NO task-oriented tag injections.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from .pair_guard import EvidenceItem, PairGuard, STOPWORDS, tokenize_words
from .jev_client import JevReranker


ENCODER = tiktoken.get_encoding("o200k_base")

CHANNEL_QUOTAS = {"event": 0.55, "episode": 0.25, "raw": 0.20}


@lru_cache(maxsize=100000)
def token_count(text: str) -> int:
    return len(ENCODER.encode(text or ""))


@dataclass
class AnchoredContextBlock:
    memory_id: str
    kind: str
    content: str
    source_time: str = ""
    role: str = ""
    parent_id: str = ""
    session_id: str = ""
    seq_id: int = 0
    score: float = 0.0
    tokens: int = 0

    def __post_init__(self):
        if self.tokens <= 0:
            self.tokens = token_count(self.rendered_text)

    @property
    def rendered_text(self) -> str:
        # For Event and Episode blocks, time/metadata are already formatted inside content
        if self.kind in {"event", "episode"}:
            return self.content
        header_parts = []
        if self.source_time:
            is_iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", self.source_time.strip()))
            has_rel = bool(re.search(
                r"\b(yesterday|last (week|month|year|friday|monday|tuesday|wednesday|thursday|saturday|sunday)|ago|next (week|month|year)|tomorrow|currently|recently|today|this morning|past)\b",
                self.content,
                re.IGNORECASE,
            ))
            if is_iso:
                header_parts.append(f"Event Date: {self.source_time}")
            elif has_rel or any(x in self.source_time.lower() for x in ["am", "pm", ":"]):
                header_parts.append(f"Conversation Time: {self.source_time}")
            else:
                header_parts.append(f"Time: {self.source_time}")
        if self.role:
            speaker_name = self.role
            m_speaker = re.match(r"^([A-Z][a-zA-Z0-9_\- ]{1,20}):\s+", self.content)
            if m_speaker:
                speaker_name = m_speaker.group(1).strip()
            header_parts.append(f"Speaker: {speaker_name}")

        header = " | ".join(header_parts)
        if header and f"Time: {self.source_time}" not in self.content:
            return f"[{header}]\n{self.content}"
        return self.content


class EvidenceAnchorEngine:
    """Translates retrieval hits into high-fidelity anchored source contexts within AML budgets."""

    def __init__(self, db_conn: sqlite3.Connection | None = None, enable_jev: bool = False):
        self.db = db_conn
        self.pair_guard = PairGuard(min_shared_keywords=2)
        self.enable_jev = enable_jev
        self.jev_reranker = JevReranker() if enable_jev else None

    @staticmethod
    def extract_focused_verbatim_span(
        raw_content: str,
        event_text: str,
        evidence_text: str = "",
        query: str = "",
        max_chars: int = 320,
    ) -> str:
        """Extract a high-signal verbatim excerpt from a long RAW turn without losing numbers/facts."""
        raw_clean = (raw_content or "").strip()
        if not raw_clean:
            return ""
        if len(raw_clean) <= max_chars:
            return raw_clean

        # Check if evidence_text appears directly in raw_clean
        if evidence_text:
            ev_needle = evidence_text.strip()
            pos = raw_clean.casefold().find(ev_needle[:60].casefold())
            if pos >= 0:
                half = max(40, (max_chars - len(ev_needle)) // 2)
                start = max(0, pos - half)
                end = min(len(raw_clean), start + max_chars)
                return raw_clean[start:end].strip()

        # Split into sentences/lines and pick the highest-overlap window
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？\n])\s+", raw_clean) if s.strip()]
        if len(sentences) <= 1:
            return raw_clean[:max_chars].strip()

        q_words = {w for w in tokenize_words(query) if w not in STOPWORDS}
        ev_words = {w for w in tokenize_words(event_text + " " + evidence_text) if w not in STOPWORDS}

        best_idx = 0
        best_score = -1.0
        for idx, sent in enumerate(sentences):
            s_words = set(tokenize_words(sent))
            if not s_words:
                continue
            score = (
                2.5 * len(s_words & q_words)
                + 1.8 * len(s_words & ev_words)
                + (1.2 if re.search(r"\d", sent) else 0.0)
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        selected = [sentences[best_idx]]
        curr_len = len(selected[0])
        # Try expanding one sentence after and one before if within max_chars
        if best_idx + 1 < len(sentences) and curr_len + 1 + len(sentences[best_idx + 1]) <= max_chars:
            selected.append(sentences[best_idx + 1])
            curr_len += 1 + len(sentences[best_idx + 1])
        if best_idx - 1 >= 0 and curr_len + 1 + len(sentences[best_idx - 1]) <= max_chars:
            selected.insert(0, sentences[best_idx - 1])

        joined = " ".join(selected).strip()
        return joined[:max_chars].strip()

    def resolve_raw_sources_for_event(self, event_item: dict, user_id: str) -> list[dict]:
        """Backward-compatible lookup for raw messages."""
        best = self.resolve_single_best_raw(event_item, user_id)
        return [best] if best else []

    def resolve_single_best_raw(self, event_item: dict, user_id: str, query: str = "", raw_cache: dict | None = None) -> dict | None:
        """Find the single most relevant RAW message that directly produced this Event."""
        event_text = str(event_item.get("content") or "")
        evidence_text = str(event_item.get("evidence") or "")
        parent_id = str(event_item.get("parent_id") or "")
        source_refs = event_item.get("source_refs") or []

        if isinstance(source_refs, str):
            try:
                source_refs = json.loads(source_refs)
            except Exception:
                source_refs = []

        # 1. First priority: exact source_ref match if available
        if source_refs:
            for ref in source_refs:
                if not ref:
                    continue
                if raw_cache:
                    hit = raw_cache.get("by_ref", {}).get(str(ref)) or raw_cache.get("by_id", {}).get(str(ref))
                    if hit:
                        return dict(hit)
                elif self.db:
                    row = self.db.execute(
                        "SELECT id, kind, content, role, source_time, source_refs FROM memories "
                        "WHERE user_id=? AND kind='raw' AND (source_refs LIKE ? OR id=?) LIMIT 1",
                        (user_id, f'%"{ref}"%', ref)
                    ).fetchone()
                    if row:
                        return dict(row)

        if not parent_id:
            return None

        req_id = ""
        t_idx = 0
        if raw_cache:
            parent_row = raw_cache.get("episodes", {}).get(parent_id)
            if not parent_row:
                return None
            req_id = parent_row["request_id"]
            t_idx = parent_row["turn_index"]
            raw_rows = [
                r for r in raw_cache.get("by_req", {}).get(req_id, [])
                if t_idx <= r.get("turn_index", 0) < t_idx + 8
            ]
        elif self.db:
            parent_row = self.db.execute(
                "SELECT turn_index, request_id, session_id FROM memories WHERE id=? LIMIT 1",
                (parent_id,)
            ).fetchone()
            if not parent_row:
                return None
            req_id = parent_row["request_id"]
            t_idx = parent_row["turn_index"]
            raw_rows = [dict(x) for x in self.db.execute(
                "SELECT id, kind, content, role, source_time, source_refs FROM memories "
                "WHERE user_id=? AND request_id=? AND kind='raw' AND turn_index>=? AND turn_index<? "
                "ORDER BY turn_index ASC",
                (user_id, req_id, t_idx, t_idx + 8)
            ).fetchall()]
        else:
            return None

        if not raw_rows:
            return None

        ev_tokens = {w for w in tokenize_words(evidence_text or event_text) if w not in STOPWORDS}
        if not ev_tokens:
            ev_tokens = set(tokenize_words(evidence_text or event_text))
        q_tokens = {w for w in tokenize_words(query) if w not in STOPWORDS}

        best_row = None
        best_overlap = -1.0

        for r in raw_rows:
            content = r["content"]
            r_tokens = set(tokenize_words(content))
            if not r_tokens:
                continue
            if evidence_text and len(evidence_text) >= 8 and (evidence_text[:48].lower() in content.lower()):
                return dict(r)
            overlap = (
                len(ev_tokens & r_tokens) / max(1, len(ev_tokens))
                + 0.25 * (len(q_tokens & r_tokens) / max(1, len(q_tokens)))
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = r

        if best_row and best_overlap > 0.12:
            return dict(best_row)

        return None

    def assemble_anchored_context(
        self,
        retrieval_hits: list[dict],
        user_id: str,
        budget: int = 8000,
        enable_pair_guard: bool = True,
        query: str = "",
        channel_quotas: dict[str, float] | None = None,
        undated_dialogue_mode: bool = False,
        bypass_jev: bool = False,
        raw_cache: dict | None = None,
    ) -> list[dict]:
        if not retrieval_hits:
            return []

        quotas = channel_quotas if channel_quotas is not None else CHANNEL_QUOTAS

        # Cache parent episode lookup and track already-rendered episode summaries
        parent_cache: dict[str, str] = {}
        rendered_episodes: set[str] = set()

        # 1. Map Events to their precision Tri-Tier representation (Event + Episode + Focused Verbatim Source)
        grounded_candidates: list[EvidenceItem] = []
        seen_ids: set[str] = set()

        for hit in retrieval_hits:
            item_id = str(hit.get("id") or "")
            if not item_id or item_id in seen_ids:
                continue
            kind = str(hit.get("kind") or ("raw" if item_id.startswith("raw_") else ("episode" if item_id.startswith("ep_") else "event")))
            score = float(hit.get("score") or 0.0)
            source_time = str(hit.get("source_time") or hit.get("time_anchor") or "")
            seq_id = int(hit.get("seq_id") or 0)

            if kind == "event":
                event_text = str(hit.get("content") or "").strip()
                if not event_text:
                    continue

                lines = []
                if source_time:
                    is_iso_date = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", source_time.strip()))
                    if is_iso_date:
                        lines.append(f"[Event Date: {source_time}]")
                    else:
                        lines.append(f"[Conversation Time: {source_time}]")
                elif undated_dialogue_mode and seq_id > 0:
                    lines.append(f"[Turn #{seq_id}]")

                lines.append(f"Event: {event_text}")

                parent_id = str(hit.get("parent_id") or "")
                parent_text = str(hit.get("parent_content") or "").strip()
                if not parent_text and parent_id:
                    if parent_id not in parent_cache:
                        if raw_cache:
                            ep_row = raw_cache.get("episodes", {}).get(parent_id)
                            parent_cache[parent_id] = str(ep_row.get("content") or "").strip() if ep_row else ""
                        elif self.db:
                            try:
                                parent_row = self.db.execute(
                                    "SELECT content FROM memories WHERE id=? LIMIT 1", (parent_id,)
                                ).fetchone()
                                parent_cache[parent_id] = str(parent_row["content"] or "").strip() if parent_row else ""
                            except Exception:
                                parent_cache[parent_id] = ""
                    parent_text = parent_cache[parent_id]

                if parent_text:
                    if undated_dialogue_mode:
                        if parent_id and parent_id not in rendered_episodes:
                            lines.append(f"Episode: {parent_text[:450]}")
                            rendered_episodes.add(parent_id)
                        else:
                            lines.append(f"Episode: {parent_text[:140]}")
                    else:
                        lines.append(f"Episode: {parent_text[:450]}")

                ev_quote = str(hit.get("evidence") or "").strip()
                focused_raw = ""
                raw_time_mismatch = False
                if self.db or raw_cache:
                    try:
                        best_raw = self.resolve_single_best_raw(hit, user_id=user_id, query=query, raw_cache=raw_cache)
                        if best_raw and best_raw.get("content"):
                            raw_st = str(best_raw.get("source_time") or "")
                            if source_time and raw_st and len(source_time) >= 10 and len(raw_st) >= 10:
                                if source_time[:10] != raw_st[:10]:
                                    raw_time_mismatch = True
                            if not raw_time_mismatch:
                                focused_raw = self.extract_focused_verbatim_span(
                                    best_raw["content"],
                                    event_text=event_text,
                                    evidence_text=ev_quote,
                                    query=query,
                                    max_chars=300,
                                )
                    except Exception:
                        focused_raw = ""

                # Combine stored evidence quote with focused raw span without bloat or date confusion
                source_line = ""
                if ev_quote and focused_raw:
                    if ev_quote[:50].casefold() in focused_raw.casefold():
                        source_line = focused_raw
                    elif focused_raw[:50].casefold() in ev_quote.casefold():
                        source_line = ev_quote[:260]
                    else:
                        source_line = f"{ev_quote[:160]} | {focused_raw[:220]}"
                elif ev_quote:
                    source_line = ev_quote[:260]
                elif focused_raw and focused_raw.casefold() != event_text.casefold():
                    source_line = focused_raw

                if source_line:
                    lines.append(f"Source: {source_line}")

                composite_content = "\n".join(lines)
                seen_ids.add(item_id)
                grounded_candidates.append(EvidenceItem(
                    id=item_id,
                    content=composite_content,
                    score=score,
                    kind="event",
                    timestamp=source_time,
                    source_ref=str(hit.get("source_refs") or ""),
                    parent_id=parent_id,
                    session_id=str(hit.get("session_id") or parent_id or ""),
                    atomic_text=event_text,
                    seq_id=seq_id,
                ))
                continue

            elif kind == "episode":
                ep_text = str(hit.get("content") or "").strip()
                if not ep_text:
                    continue
                content = f"[Session Time: {source_time}]\n{ep_text}" if source_time else ep_text
                seen_ids.add(item_id)
                grounded_candidates.append(EvidenceItem(
                    id=item_id,
                    content=content,
                    score=score,
                    kind="episode",
                    timestamp=source_time,
                    source_ref=str(hit.get("source_refs") or ""),
                    parent_id=str(hit.get("parent_id") or ""),
                    session_id=str(hit.get("session_id") or ""),
                    seq_id=seq_id,
                ))
                continue

            # Direct RAW item
            raw_text = str(hit.get("content") or "").strip()
            if not raw_text:
                continue
            role_str = str(hit.get("role") or "").strip()
            focused_raw_atomic = ""
            if undated_dialogue_mode:
                focused_raw_atomic = self.extract_focused_verbatim_span(
                    raw_text, event_text="", evidence_text="", query=query, max_chars=260
                )
            raw_headers = []
            if source_time:
                raw_headers.append(f"Conversation Time: {source_time}")
            elif undated_dialogue_mode and seq_id > 0:
                raw_headers.append(f"Turn #{seq_id}")
            if role_str:
                speaker_name = role_str
                m_speaker = re.match(r"^([A-Z][a-zA-Z0-9_\- ]{1,20}):\s+", raw_text)
                if m_speaker:
                    speaker_name = m_speaker.group(1).strip()
                raw_headers.append(f"Speaker: {speaker_name}")
            if raw_headers:
                raw_text = f"[{' | '.join(raw_headers)}]\n{raw_text}"
            seen_ids.add(item_id)
            grounded_candidates.append(EvidenceItem(
                id=item_id,
                content=raw_text,
                score=score,
                kind="raw",
                timestamp=source_time,
                source_ref=str(hit.get("source_refs") or ""),
                parent_id=str(hit.get("parent_id") or ""),
                session_id=str(hit.get("session_id") or hit.get("parent_id") or ""),
                atomic_text=focused_raw_atomic,
                seq_id=seq_id,
            ))

        # 2. Apply Pair Guard within candidate pool to protect temporal updates and contradictions
        detected_pairs = []
        if enable_pair_guard:
            guarded_items, detected_pairs = self.pair_guard.apply_pair_guard(grounded_candidates)
        else:
            guarded_items = grounded_candidates

        # 3. Deduplicate and build AnchoredContextBlocks
        context_blocks: list[AnchoredContextBlock] = []
        seen_contents: set[str] = set()

        for ev in guarded_items:
            norm_c = " ".join(re.findall(r"[\w]+", ev.content.casefold()))
            if not norm_c or norm_c in seen_contents:
                continue
            seen_contents.add(norm_c)

            context_blocks.append(AnchoredContextBlock(
                memory_id=ev.id,
                kind=ev.kind,
                content=ev.content,
                source_time=ev.timestamp,
                parent_id=ev.parent_id,
                session_id=str(getattr(ev, "session_id", "") or ev.parent_id),
                seq_id=int(getattr(ev, "seq_id", 0) or 0),
                score=ev.score,
            ))

        # Build pair partner lookup and relation type lookup for Atomic Pair Admission & JEV Synergy
        pair_partner_map: dict[str, str] = {}
        pair_relation_map: dict[str, str] = {}
        for p in detected_pairs:
            pair_partner_map[p.item_a.id] = p.item_b.id
            pair_partner_map[p.item_b.id] = p.item_a.id
            pair_relation_map[p.item_a.id] = p.relation_type
            pair_relation_map[p.item_b.id] = p.relation_type

        # JEV Pointwise Semantic Reranking on Top-36 Candidates (Hybrid Residual Layer)
        if self.enable_jev and not bypass_jev and self.jev_reranker and self.jev_reranker.key and query:
            top_candidates = context_blocks[:36]
            contents = [b.content for b in top_candidates]
            jev_scores = self.jev_reranker.score_batch(query, contents)
            block_by_id_temp = {b.memory_id: b for b in context_blocks}
            for b, js in zip(top_candidates, jev_scores):
                norm_js = js / 3.0
                b.score = b.score + 0.45 * norm_js
                # Pair-synergy propagation: ensure pair partner travels with high-scoring item
                partner_id = pair_partner_map.get(b.memory_id)
                if partner_id and partner_id in block_by_id_temp:
                    p_block = block_by_id_temp[partner_id]
                    p_block.score = max(p_block.score, b.score - 0.05)
            context_blocks.sort(key=lambda b: (-b.score, b.memory_id))

        block_by_id = {b.memory_id: b for b in context_blocks}

        # 4. Tri-Tier Channel-Structured Budget Allocation + Atomic Pair Admission + Diversity
        quotas = channel_quotas or CHANNEL_QUOTAS
        usable_budget = max(200, budget - 64)
        selected_blocks: list[AnchoredContextBlock] = []
        used_ids: set[str] = set()
        current_tokens = 0
        ep_event_counts: dict[str, int] = {}

        is_enumeration = bool(re.search(
            r"\b(how many|list all|what are the|order of|walk me through|summarize all|which|each|every)\b",
            (query or "").casefold(),
        ))
        max_events_primary = 1 if is_enumeration else (2 if undated_dialogue_mode else 4)

        for channel in ("event", "episode", "raw"):
            target_tokens = max(1, int(usable_budget * quotas.get(channel, 0.0)))
            channel_tokens = 0
            for block in context_blocks:
                if block.memory_id in used_ids or block.kind != channel:
                    continue
                if channel == "event" and block.parent_id:
                    if ep_event_counts.get(block.parent_id, 0) >= max_events_primary:
                        continue
                if channel_tokens >= target_tokens:
                    break

                partner_id = pair_partner_map.get(block.memory_id)
                partner_block = block_by_id.get(partner_id) if partner_id else None

                # Atomic Pair Admission: if block has an unadmitted partner in the same channel, admit together or neither
                if partner_block and partner_block.kind == channel and partner_block.memory_id not in used_ids:
                    pair_cost = (block.tokens + 6) + (partner_block.tokens + 6)
                    if current_tokens + pair_cost <= usable_budget:
                        selected_blocks.extend([block, partner_block])
                        used_ids.add(block.memory_id)
                        used_ids.add(partner_block.memory_id)
                        channel_tokens += (block.tokens + partner_block.tokens)
                        current_tokens += pair_cost
                        if channel == "event":
                            if block.parent_id:
                                ep_event_counts[block.parent_id] = ep_event_counts.get(block.parent_id, 0) + 1
                            if partner_block.parent_id:
                                ep_event_counts[partner_block.parent_id] = ep_event_counts.get(partner_block.parent_id, 0) + 1
                else:
                    # Single block admission
                    block_cost = block.tokens + 6
                    if current_tokens + block_cost <= usable_budget:
                        selected_blocks.append(block)
                        used_ids.add(block.memory_id)
                        channel_tokens += block.tokens
                        current_tokens += block_cost
                        if channel == "event" and block.parent_id:
                            ep_event_counts[block.parent_id] = ep_event_counts.get(block.parent_id, 0) + 1

        # Relevance backfill for any unused budget (preserving atomic pair guarantee)
        for block in context_blocks:
            if block.memory_id in used_ids:
                continue
            if block.kind == "event" and block.parent_id:
                if ep_event_counts.get(block.parent_id, 0) >= (max_events_primary + 2):
                    continue
            if current_tokens >= usable_budget:
                break

            partner_id = pair_partner_map.get(block.memory_id)
            partner_block = block_by_id.get(partner_id) if partner_id else None

            if partner_block and partner_block.memory_id not in used_ids:
                pair_cost = (block.tokens + 6) + (partner_block.tokens + 6)
                if current_tokens + pair_cost <= usable_budget:
                    selected_blocks.extend([block, partner_block])
                    used_ids.add(block.memory_id)
                    used_ids.add(partner_block.memory_id)
                    current_tokens += pair_cost
                    if block.kind == "event" and block.parent_id:
                        ep_event_counts[block.parent_id] = ep_event_counts.get(block.parent_id, 0) + 1
                    if partner_block.kind == "event" and partner_block.parent_id:
                        ep_event_counts[partner_block.parent_id] = ep_event_counts.get(partner_block.parent_id, 0) + 1
            else:
                block_cost = block.tokens + 6
                if current_tokens + block_cost <= usable_budget:
                    selected_blocks.append(block)
                    used_ids.add(block.memory_id)
                    current_tokens += block_cost
                    if block.kind == "event" and block.parent_id:
                        ep_event_counts[block.parent_id] = ep_event_counts.get(block.parent_id, 0) + 1

        # Preserve relevance + PairGuard ordering across the quota-selected blocks
        order_index = {b.memory_id: idx for idx, b in enumerate(context_blocks)}
        selected_blocks.sort(key=lambda b: order_index.get(b.memory_id, 999999))

        output_records = []
        for b in selected_blocks:
            rec = {
                "id": b.memory_id,
                "content": b.rendered_text,
                "score": round(b.score, 6),
                "created_at": b.source_time,
            }
            p_id = pair_partner_map.get(b.memory_id)
            if p_id:
                rec["partner_id"] = p_id
            output_records.append(rec)

        return output_records
