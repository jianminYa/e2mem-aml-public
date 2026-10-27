"""Contradiction and Temporal Update Pair Guard for E2Mem.

Ensures contradictory statements or temporal state progressions regarding the
same entity/subject are preserved as paired evidence in the retrieved context,
preventing one-sided semantic crowding from dropping critical counter-evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# Polarity and negation cues
NEGATION_WORDS = frozenset({
    "never", "not", "no", "n't", "none", "neither", "nor", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "stopped", "quit", "ceased",
    "dislike", "hate", "refused", "cancelled", "canceled", "rejected",
    "false", "untrue", "incorrect"
})

# Future / planning / modal cues vs completed / current status cues
PLANNING_CUES = frozenset({
    "plan", "plans", "planning", "planned", "intend", "intends", "scheduled",
    "upcoming", "future", "hoping", "hope", "will", "might", "considering"
})

COMPLETED_CUES = frozenset({
    "already", "completed", "done", "finished", "currently", "now", "today",
    "yesterday", "visited", "bought", "moved", "attended", "arrived"
})

UPDATE_CUES = frozenset({
    "extended", "increased", "decreased", "changed", "updated", "switched",
    "moved", "replaced", "reduced", "raised", "adjusted", "postponed",
    "delayed", "rescheduled", "pushed", "instead", "anymore", "surpassed"
})

STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "been", "be", "have",
    "has", "had", "do", "does", "did", "to", "at", "in", "on", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "of",
    "and", "but", "or", "as", "if", "because", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "their", "this", "that", "these", "those", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "would", "could", "might", "may", "also"
})

CONV_STOPWORDS = frozenset({
    "keep", "keeps", "keeping", "kept",
    "said", "say", "says", "saying",
    "told", "tell", "tells", "telling",
    "talk", "talks", "talking", "talked",
    "feel", "feels", "feeling", "felt",
    "think", "thinks", "thinking", "thought",
    "know", "knows", "knowing", "knew",
    "going", "goes", "went", "gone",
    "come", "comes", "coming", "came",
    "make", "makes", "making", "made",
    "take", "takes", "taking", "took",
    "good", "nice", "well", "great", "sure",
    "time", "day", "week", "month", "year",
    "user", "assistant", "system", "event", "episode",
    "recent", "conversation", "expressed", "shared", "stated",
    "reflected", "noted", "mentioned"
})



def normalize_contractions(text: str) -> str:
    """Normalize English contractions like didn't -> did not."""
    return re.sub(r"\b(\w+)n['’]t\b", r"\1 not", text, flags=re.IGNORECASE)


def tokenize_words(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens with normalized contractions."""
    text_norm = normalize_contractions(text)
    return re.findall(r"[\w]+|[\u3400-\u9fff]", text_norm.casefold(), re.UNICODE)


def has_negation(tokens: Sequence[str]) -> bool:
    """Check if token sequence contains explicit negation markers."""
    return any(tok in NEGATION_WORDS or tok.endswith("n't") for tok in tokens)


def planning_status(tokens: Sequence[str]) -> str:
    """Classify statement into planned, completed, or neutral."""
    has_plan = any(tok in PLANNING_CUES for tok in tokens)
    has_done = any(tok in COMPLETED_CUES for tok in tokens)
    if has_plan and not has_done:
        return "planned"
    if has_done and not has_plan:
        return "completed"
    return "neutral"


@dataclass
class EvidenceItem:
    id: str
    content: str
    score: float
    kind: str  # 'event', 'episode', 'raw'
    timestamp: str = ""
    source_ref: str = ""
    parent_id: str = ""
    session_id: str = ""
    atomic_text: str = ""
    seq_id: int = 0
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = tokenize_words(self.atomic_text or self.content)


@dataclass
class PairedRelation:
    item_a: EvidenceItem
    item_b: EvidenceItem
    relation_type: str  # 'contradiction' | 'temporal_update'
    shared_keywords: frozenset[str]
    confidence: float


class PairGuard:
    """Detects and protects paired contradictions and temporal progressions."""

    def __init__(self, min_shared_keywords: int = 2, stopword_cutoff: int = 100):
        self.min_shared_keywords = min_shared_keywords
        self.stopwords = STOPWORDS | CONV_STOPWORDS

    def content_keywords(self, tokens: Sequence[str]) -> frozenset[str]:
        return frozenset(t for t in tokens if len(t) > 2 and t not in self.stopwords and t not in NEGATION_WORDS)

    def detect_pairs(self, items: list[EvidenceItem]) -> list[PairedRelation]:
        """Scan a candidate pool and identify paired contradiction or temporal update candidates."""
        pairs: list[PairedRelation] = []
        n = len(items)
        if n == 0:
            return pairs
        top_score = max((it.score for it in items), default=1.0)
        min_admit_score = max(0.35, 0.40 * top_score)

        # Precompute per-item lexical and semantic attributes to avoid repeated O(N^2) evaluation
        item_props = []
        for it in items:
            kws = self.content_keywords(it.tokens)
            neg = has_negation(it.tokens)
            stat = planning_status(it.tokens)
            nums = frozenset(t for t in it.tokens if any(ch.isdigit() for ch in t))
            non_year = frozenset(t for t in nums if not (len(t) == 4 and t.startswith(("19", "20"))))
            has_update = any(t in UPDATE_CUES for t in it.tokens)
            item_props.append((kws, neg, stat, nums, non_year, has_update))

        for i in range(n):
            item_a = items[i]
            if item_a.score < min_admit_score:
                continue
            kw_a, neg_a, status_a, nums_a, non_year_a, has_update_a = item_props[i]
            if len(kw_a) < self.min_shared_keywords:
                continue

            for j in range(i + 1, n):
                item_b = items[j]
                # Only pair items within the same channel/kind and with meaningful relevance
                if item_a.kind != item_b.kind:
                    continue
                if item_b.score < min_admit_score:
                    continue
                # Avoid pairing sibling events from the exact same episode
                if item_a.parent_id and item_b.parent_id and item_a.parent_id == item_b.parent_id:
                    continue
                # Avoid pairing duplicate copies of the same item
                if item_a.id == item_b.id or item_a.content.strip() == item_b.content.strip():
                    continue

                kw_b, neg_b, status_b, nums_b, non_year_b, has_update_b = item_props[j]
                shared = kw_a & kw_b
                if len(shared) < self.min_shared_keywords:
                    continue

                overlap_ratio = len(shared) / max(1, min(len(kw_a), len(kw_b)))
                if overlap_ratio < 0.25:
                    continue

                has_update_pair = has_update_b or has_update_a

                # Case 1: Contradiction (opposite polarity on shared subject)
                if neg_a != neg_b:
                    confidence = 0.5 + 0.5 * min(1.0, len(shared) / 4.0)
                    pairs.append(PairedRelation(
                        item_a=item_a,
                        item_b=item_b,
                        relation_type="contradiction",
                        shared_keywords=shared,
                        confidence=confidence,
                    ))
                    continue

                # Case 2: Temporal state progression (planned vs completed) or explicit update/numerical counterpart
                is_plan_done = (
                    (status_a == "planned" and status_b == "completed")
                    or (status_a == "completed" and status_b == "planned")
                ) and (len(shared) >= 3 or overlap_ratio >= 0.40)
                has_num_conflict = (
                    bool(nums_a and nums_b and nums_a != nums_b)
                    and not (non_year_a and non_year_a == non_year_b)
                )
                is_num_or_update = (
                    len(shared) >= 3
                    and overlap_ratio >= 0.38
                    and (has_update_pair or has_num_conflict)
                )
                if is_plan_done or is_num_or_update:
                    confidence = 0.4 + 0.4 * min(1.0, len(shared) / 4.0)
                    pairs.append(PairedRelation(
                        item_a=item_a,
                        item_b=item_b,
                        relation_type="temporal_update",
                        shared_keywords=shared,
                        confidence=confidence,
                    ))

        return pairs

    def apply_pair_guard(
        self,
        ranked_items: list[EvidenceItem],
        max_promotions: int = 5,
    ) -> tuple[list[EvidenceItem], list[PairedRelation]]:
        """Re-sort items such that if a primary item is selected, its counterpart
        (contradiction or update) is prioritized and admitted alongside it.
        """
        pairs = self.detect_pairs(ranked_items)
        if not pairs:
            return list(ranked_items), []

        # Map each item id to its partner items and relation types
        partner_map: dict[str, list[tuple[EvidenceItem, str]]] = {}
        for p in pairs:
            partner_map.setdefault(p.item_a.id, []).append((p.item_b, p.relation_type))
            partner_map.setdefault(p.item_b.id, []).append((p.item_a, p.relation_type))

        reordered: list[EvidenceItem] = []
        admitted_ids: set[str] = set()
        promotions_done = 0

        for item in ranked_items:
            if item.id in admitted_ids:
                continue

            # Check if this item has a paired counter-evidence partner that is ranked lower
            partners = partner_map.get(item.id, [])
            promoted_before: list[EvidenceItem] = []
            promoted_after: list[EvidenceItem] = []

            for partner, rel_type in partners:
                if partner.id not in admitted_ids and promotions_done < max_promotions:
                    is_newer_undated = (
                        rel_type == "temporal_update"
                        and not partner.timestamp
                        and not item.timestamp
                        and partner.seq_id > item.seq_id > 0
                    )
                    if is_newer_undated:
                        promoted_before.append(partner)
                    else:
                        promoted_after.append(partner)
                    admitted_ids.add(partner.id)
                    promotions_done += 1

            for p_item in promoted_before:
                reordered.append(p_item)

            reordered.append(item)
            admitted_ids.add(item.id)

            for p_item in promoted_after:
                reordered.append(p_item)

        return reordered, pairs
