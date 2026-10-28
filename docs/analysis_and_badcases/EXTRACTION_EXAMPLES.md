# Real-World Extraction Artifacts & Tri-Tier Representations

> **Document**: Actual Extraction Pipeline Traces  
> **Repository**: E²-Mem Evidence Anchor (v2.6.1)  
> **Source**: AML Multi-Session Conversational & Document Ingestion Traces  

This document illustrates the actual structured memory units produced by E²-Mem during the ingestion phase (`POST /add`). It demonstrates how raw multi-turn dialogue is transformed into our **Tri-Tier Hierarchy** (Atomic Events, Narrative Episodes, and Verbatim Raw Turns).

---

## 1. Dialogue Ingestion Example (LoCoMo Transcript)

### 1.1 Input Raw Transcript (`POST /add`)
```json
{
  "request_id": "req_demo_locomo_turn_42",
  "user_id": "conv-26",
  "session_id": "session_3",
  "messages": [
    {
      "role": "user",
      "content": "Caroline: Sounds great, Mel! Glad you had a great time. Cherish the moments - they're the best!",
      "source_ref": "D3:21",
      "timestamp": "7:55 pm on 9 June, 2023"
    },
    {
      "role": "assistant",
      "content": "Melanie: Absolutely, Caroline! I cherish time with family. It's when I really feel alive and happy.",
      "source_ref": "D3:22",
      "timestamp": "7:55 pm on 9 June, 2023"
    },
    {
      "role": "user",
      "content": "Caroline: I 100% agree, Mel. Hanging with loved ones is amazing and brings so much happiness. Those moments really make me thankful. Family is everything.",
      "source_ref": "D3:23",
      "timestamp": "7:55 pm on 9 June, 2023"
    }
  ]
}
```

---

### 1.2 Extracted Tri-Tier Memory Representations

#### Tier 1: Atomic Events (Fine-grained Facts with Timestamps)
These are granular propositions with resolved timestamps, keywords, and entity aliases:
```json
[
  {
    "id": "ev_81f0d3a9",
    "kind": "event",
    "event_text": "Melanie expresses deep appreciation for spending time with family, stating it brings happiness and vitality.",
    "evidence": [
      "Melanie: Absolutely, Caroline! I cherish time with family. It's when I really feel alive and happy."
    ],
    "aliases": ["Melanie", "Mel"],
    "keywords": ["family", "happiness", "cherish moments", "vitality"],
    "fact_date_resolved": "2023-06-09",
    "event_type": "preference",
    "session_id": "session_3"
  },
  {
    "id": "ev_81f0d3aa",
    "kind": "event",
    "event_text": "Caroline agrees that spending time with family brings immense happiness and expresses gratitude.",
    "evidence": [
      "Caroline: I 100% agree, Mel. Hanging with loved ones is amazing and brings so much happiness. Those moments really make me thankful. Family is everything."
    ],
    "aliases": ["Caroline"],
    "keywords": ["family", "loved ones", "gratitude", "happiness"],
    "fact_date_resolved": "2023-06-09",
    "event_type": "preference",
    "session_id": "session_3"
  }
]
```

#### Tier 2: Narrative Episodes (Thematic Summaries & High-Level Context)
Provides broad thematic grounding and emotional context:
```json
{
  "id": "ep_38d9f10a",
  "kind": "episode",
  "subject": "Discussion on family values and cherishing meaningful moments",
  "summary": "Caroline and Melanie discuss the profound joy of spending time with family, sharing mutual agreement that family connections are essential for happiness and gratitude.",
  "topic_labels": [
    "family",
    "emotional well-being",
    "gratitude",
    "cherishing moments"
  ],
  "session_id": "session_3",
  "timestamp": "2023-06-09T19:55:00Z"
}
```

#### Tier 3: Verbatim Raw Turns (Ground-Truth Literal Dialogue)
Indexed with exact utterance timestamps for literal grounding:
```json
[
  {
    "id": "raw_91b72a01",
    "kind": "raw",
    "content": "[Conversation Time: 2023-06-09T19:55:00Z | Speaker: user]\nCaroline: Sounds great, Mel! Glad you had a great time. Cherish the moments - they're the best!",
    "source_ref": "D3:21"
  },
  {
    "id": "raw_91b72a02",
    "kind": "raw",
    "content": "[Conversation Time: 2023-06-09T19:55:00Z | Speaker: assistant]\nMelanie: Absolutely, Caroline! I cherish time with family. It's when I really feel alive and happy.",
    "source_ref": "D3:22"
  }
]
```

---

## 2. Document Artifact Ingestion Example (CLBench Statutory Artifact)

For long statutory and guideline documents (e.g. municipal ordinances, clinical diagnostic criteria):

### Modality Detection
- Physical detector detects `avg_raw_len >= 2400` or `max_len >= 6000 and count <= 15`.
- Classified as `document_artifact`.
- Rather than slicing into arbitrary sentence chunks that separate rules from their exceptions, the document is ingested with its section hierarchy preserved, triggering the adaptive **24K token context budget**.
