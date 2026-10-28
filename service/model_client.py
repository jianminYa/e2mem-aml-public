"""Robust HTTP ModelClient for AML evaluation."""
import json
import os
import time
import urllib.error
import urllib.request
try:
    import httpx
except ImportError:
    httpx = None


class ModelClient:
    def __init__(self, mode: str = "simulator"):
        self.mode = mode
        self.url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.embedding_url = os.getenv("E2MEM_EMBEDDING_BASE_URL", self.url).rstrip("/")
        self.embedding_key = os.getenv("E2MEM_EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or self.key
        self.chat_model = os.getenv("E2MEM_CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("E2MEM_EMBEDDING_MODEL", "text-embedding-v4")
        timeout = float(os.environ.get("E2MEM_REQUEST_TIMEOUT_SECONDS", "30"))
        if httpx:
            self._client = httpx.Client(
                http2=True,
                timeout=timeout,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        else:
            self._client = None

    def request(self, endpoint: str, payload: dict, *, embedding: bool = False) -> dict:
        target_url = (self.embedding_url if embedding else self.url) + endpoint
        target_key = self.embedding_key if embedding else self.key
        headers = {"Authorization": "Bearer " + target_key, "Content-Type": "application/json"}
        timeout = int(os.environ.get("E2MEM_REQUEST_TIMEOUT_SECONDS", "30"))

        for attempt in range(4):
            if self._client:
                try:
                    resp = self._client.post(target_url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        return resp.json()
                    if attempt == 3:
                        raise urllib.error.HTTPError(
                            target_url, resp.status_code, f"HTTP {resp.status_code}", {}, None
                        )
                except urllib.error.HTTPError:
                    raise
                except Exception as err:
                    if attempt == 3:
                        raise
                time.sleep(min(8, 2 ** attempt))
                continue

            req = urllib.request.Request(
                target_url,
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise
            except urllib.error.URLError:
                if attempt == 3:
                    raise
            time.sleep(min(8, 2 ** attempt))
        raise RuntimeError("provider retry loop exhausted")

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        if not texts or self.mode == "simulator":
            return [[] for _ in texts], {"calls": 0, "tokens": 0}
        vectors, calls, total_tokens = [], 0, 0
        for start in range(0, len(texts), 10):
            data = self.request(
                "/embeddings",
                {"model": self.embedding_model, "input": texts[start:start + 10]},
                embedding=True,
            )
            vectors.extend(item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"]))
            calls += 1
            total_tokens += (data.get("usage") or {}).get("total_tokens", 0)
        if len(vectors) != len(texts):
            raise RuntimeError("embedding response size mismatch")
        return vectors, {"calls": calls, "tokens": total_tokens}

    def extract(self, messages: list[dict], session_id: str, user_id: str = "", prior_context: list[dict] | None = None) -> tuple[dict, dict]:
        import re
        text = " ".join(m.get("content", "") for m in messages)
        if self.mode == "simulator":
            clauses = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*", text) if s.strip()]
            return ({
                "subject": "Conversation",
                "summary": text[:500],
                "episode": text[:1200],
                "topic_labels": [],
                "episode_events": [
                    {"event_text": s, "evidence": [s], "aliases": [], "keywords": [], "fact_date_resolved": ""}
                    for s in clauses[:10]
                ]
            }, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

        # Competition mode: Extraction v3.0 High-Fidelity Multi-Facet Extractor
        system_prompt = (
            "You are an expert factual memory and policy rule extraction engine for long-term agentic memory.\n"
            "Extract all atomic factual propositions, procedural rules, constraints, user acquisitions/investments, "
            "preferences, and state changes from the input text.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. DYNAMIC DENSITY: Extract ALL concrete facts, rules, and statements without arbitrarily omitting details or capping at 5. "
            "Typically extract 3 to 15 events depending on information density.\n"
            "2. HIGH-FIDELITY ENTITIES: Preserve exact titles (in quotation marks, e.g. \"Principles of Computing\"), specific work/article/media names, "
            "device/hardware/tool model names (e.g. 'Smart Thermostat', 'Digital Multimeter'), numbers, measurements, and proper nouns. "
            "Never replace a specific title with a generic term like 'an item'.\n"
            "3. ACTIONS, ACQUISITIONS & LIFECYCLE: Explicitly extract acquisitions, investments, purchases, or new items "
            "(e.g. 'User acquired a new workstation terminal'), not just secondary background discussions.\n"
            "4. RULES, PROCEDURES & CONSTRAINTS: For procedural texts, guidelines, or manuals, extract exact priority tiers (CRITICAL, HIGH, MEDIUM), "
            "response timeframes (e.g. 1 hour, 4 hours, 24 hours), expert roles (e.g. 'Lead Systems Architect', 'Certified Information Auditor', 'Project Compliance Officer'), "
            "procedural steps, and out-of-scope routing rules.\n"
            "5. CONCEPTUAL HYPERNYMS & SYNONYMS: In keywords, ALWAYS include natural category hypernyms and synonyms "
            "(e.g. for any technical hardware or device, include 'hardware', 'equipment', 'instrument', 'device'; "
            "for any purchase/acquisition, include 'invest', 'investment', 'bought', 'purchase', 'acquired'; "
            "for publications/texts, include 'document', 'publication', 'reference', 'manual').\n"
            "6. BOUNDARY & ABSTENTION AWARENESS: If an event, discussion, or feedback is mentioned but specific content/details were NOT disclosed "
            "(e.g. 'received project milestone review on October 15' without detailing the feedback), state the fact and explicitly note: "
            "'[specific feedback details were not disclosed in the text]', setting unspecified_details: true.\n"
            "7. SPEAKER ATTRIBUTION: Clearly attribute statements and actions to the respective speaker or role.\n\n"
            "Output strictly valid JSON with keys:\n"
            "- subject: string (core subject/topic)\n"
            "- summary: string (concise narrative overview)\n"
            "- episode: string (rich context description)\n"
            "- topic_labels: list of strings\n"
            "- episode_events: list of dicts, each with:\n"
            "  - event_text: string (concise, factual, self-contained proposition with exact entity names and action verbs)\n"
            "  - evidence: list of strings (verbatim text excerpts)\n"
            "  - aliases: list of strings (alternative names, entities mentioned)\n"
            "  - keywords: list of strings (specific entities, category hypernyms, dates, numbers)\n"
            "  - fact_date_resolved: string (date if mentioned or contextual, else '')\n"
            "  - event_type: string ('acquisition'|'rule'|'action'|'preference'|'constraint'|'life_event'|'boundary')\n"
            "  - unspecified_details: boolean (true if details were omitted in text, else false)\n"
        )

        user_content = ""
        if prior_context:
            user_content += f"Prior background context (for reference only, do not extract events from these):\n{json.dumps(prior_context, ensure_ascii=False)}\n\n"
        user_content += f"Target messages to extract from:\nUser: {user_id}, Session: {session_id}\n\nMessages:\n{json.dumps(messages, ensure_ascii=False)}"

        data = self.request("/chat/completions", {
            "model": self.chat_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        })
        parsed = json.loads(data["choices"][0]["message"]["content"])
        usage = data.get("usage") or {}
        return parsed, {"calls": 1, "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}


