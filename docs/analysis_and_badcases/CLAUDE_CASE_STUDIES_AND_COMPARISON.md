# Comprehensive Benchmark Case Studies & Experimental Comparisons

> **Target Audience**: AI Architecture Reviewers & Strategic Consultants (e.g., Claude)  
> **Source Data**: 248-Question Balanced AML Benchmark Cohort  
> **Systems Compared**:
> 1. `AML Baseline 1` (Official AML baseline)
> 2. `RelayRouter JEV` (Joint Evidence Verification with full candidate cross-encoder)
> 3. `Anchor v2.6.1` (Our current best submission, strict P95 < 200ms)
> **Companion Dataset**: Complete 248-question raw trace available at [`benchmarks/evaluation_results/benchmark_248_full_traces.jsonl`](../../benchmarks/evaluation_results/benchmark_248_full_traces.jsonl).

---

## 1. Overview of Experimental Head-to-Head Trends

Across the 248 balanced questions:
- **Anchor v2.6.1**: Highest overall Macro score (**0.4697**), P95 search latency **~185ms**.
- **RelayRouter JEV**: Macro score **0.4633**, but search latency exploded to **450–850ms** (violating the P95 < 200ms SLA).
- **Direct Wins**:
  - Anchor beat JEV on **26 questions** (especially long documents in CLBench and synthesis in BEAM).
  - JEV beat Anchor on **20 questions** (primarily in PersonaMem and entity disambiguation in LoCoMo).

Below, we detail the most instructive real cases from the benchmark across both failure modes and JEV/Anchor divergence.

---

## 2. Case Study A: Speaker Attribution & Entity Disambiguation (JEV Win vs. Anchor Loss)

### Metadata
- **Question ID**: `conv-49:7`
- **Dataset**: `LoCoMo` (Split: `dev_cohort`, Category: `1`)
- **User Query**: `"What new hobbies did Sam consider trying?"`
- **Scores**:
  - AML Baseline 1: `0.0`
  - **RelayRouter JEV**: `1.0` (Won)
  - **Anchor v2.6.1**: `0.0` (Failed)

### Retrieved Memory Evidence (Anchor v2.6.1 Top Hits)
```text
[Hit 1 | Score: 1.2661]
[Event Date: 2023-05-18]
Event: Sam and Evan plan to catch up soon to discuss new hobbies.
Episode: In a recent conversation, I expressed my desire to find a hobby that excites me as much as watercolor painting does for Evan. He encouraged me to keep trying new things until I discover my passion. I'm looking forward to exploring different activities and seeing what sparks my interest. Evan's enthusiasm for my journey is motivating, and we plan to catch up soon to discuss my experiences with new hobbies.
Source: Evan: No worries

[Hit 2 | Score: 0.9930]
[Conversation Time: 2023-05-18T00:00:00Z | Speaker: assistant]
Evan: Hey Sam! How's it going? Have you found any hobbies that catch your eye yet? I'm always curious to see what other people get into.

[Hit 3 | Score: 0.9248]
[Conversation Time: 2023-05-18T00:00:00Z | Speaker: user]
Sam: Hey Evan! Not yet, but I'm keeping an open mind. It's tough trying to find something that sparks that same passion for me that watercolor painting does for you.
```

### Generated Answers
- **Anchor v2.6.1 (`gpt-4o-mini`)**:
  > *"Sam considered trying painting and kayaking as new hobbies."* (Score: 0.0)
- **RelayRouter JEV (`gpt-4o-mini`)**:
  > *"Sam has not settled on specific hobbies yet, but is keeping an open mind to find a passion similar to Evan's passion for watercolor painting."* (Score: 1.0)

### Diagnostic Analysis & Question for Claude
1. **Root Cause of Anchor Failure**: In Hit 1, the text mentioned *"as watercolor painting does for Evan"*. The downstream reader (`gpt-4o-mini`) misattributed Evan's hobby (*painting*) to Sam.
2. **Why JEV Succeeded**: JEV's cross-encoder evaluated the joint proposition $(Q, \text{Hit})$ and downweighted hits where the subject entity was Evan rather than Sam, allowing the reader to recognize that Sam hadn't chosen a hobby yet.
3. **Question for Claude**: Can we achieve this entity-subject verification **within < 30ms** (e.g. lightweight subject-predicate matching or token-level late interaction) without deploying a multi-hundred-millisecond cross-encoder?

---

## 3. Case Study B: The CLBench In-Context Paradox (Complete Text Present, Zero Score)

### Metadata
- **Question ID**: `ca296b9c-ca62-42fe-b586-ba972e09ae22`
- **Dataset**: `CLBench` (Split: `holdout_cohort`)
- **User Query**: Detailed regulatory question regarding compliance exceptions under municipal building safety codes.
- **Scores**:
  - AML Baseline 1: `0.0`
  - RelayRouter JEV: `0.0`
  - Anchor v2.6.1: `0.0`

### Retrieved Memory Evidence
- **Hit 1**: Raw statutory artifact containing **6,124 characters** (~1,500 tokens).
- **Context Status**: 100% of the relevant administrative ordinance was present in the prompt. Zero retrieval truncation occurred.

### Downstream Generated Answer (`gpt-4o-mini`)
> The model generated a structured 4-paragraph response citing the main building safety requirement, but **omitted subsection 4(b)(ii) regarding historical landmark retrofitting exceptions**.

### Rubric Checklist (Binary All-or-Nothing Scoring)
- Criterion 1: Mentions building safety code section. (Passed)
- Criterion 2: Mentions structural load requirements. (Passed)
- Criterion 3: Mentions inspection timelines. (Passed)
- ...
- Criterion 11: **Must explicitly state the historical landmark exception clause.** (Failed)
- **Official Score**: `0.0` (Because $11/12 \ne 12/12$, binary rubric gives 0).

### Diagnostic Analysis & Question for Claude
- **The Dilemma**: Standard retrieval optimization cannot help here because the text is already 100% in context.
- **Question for Claude**: How can the memory service organize or format the injected statutory context (e.g., adding an auto-generated Table of Contents, extracting a "Conditions & Exceptions Checklist", or structuring clauses hierarchically) so that `gpt-4o-mini` systematically addresses all nested requirements without hallucinating?

---

## 4. Case Study C: Relative Date Arithmetic in LoCoMo

### Metadata
- **Question ID**: `conv-41:63`
- **Dataset**: `LoCoMo` (Split: `dev_cohort`)
- **User Query**: `"How many weeks passed between Maria adopting Coco and Shadow?"`
- **Scores**:
  - Baseline 1: `0.0`
  - JEV: `0.0`
  - Anchor v2.6.1: `0.0`

### Retrieved Memory Evidence
```text
[Hit 1 | Rank 1 | Score: 1.184]
[Event Date: 2023-07-28]
Event: Maria got a new puppy named Coco two weeks ago.
Episode: Maria excitedly told John about her new puppy, Coco, whom she got two weeks ago.

[Hit 2 | Rank 2 | Score: 0.901]
[Event Date: 2023-08-06]
Event: Maria adopted a puppy named Shadow from a shelter.
Episode: Maria talked about adopting a new puppy named Shadow from a shelter last week.

[Hit 3 | Rank 3] [Conversation Time: 2023-08-11T00:10:00Z] Maria: "...got a puppy two weeks ago! Her name's Coco"
[Hit 4 | Rank 4] [Conversation Time: 2023-08-13T15:14:00Z] Maria: "Her name is Shadow! She's full of energy..."
```

### Generated Answer (`gpt-4o-mini`)
> *"Three weeks passed between Maria adopting Coco and Shadow."* (Score: 0.0)

### Diagnostic Analysis & Question for Claude
1. **Mathematical Reality**:
   - Utterance 1 on August 11 says "two weeks ago" $\implies$ ~July 28.
   - Utterance 2 on August 13 says "last week" $\implies$ ~August 6.
   - The elapsed time between July 28 and August 6 is **9 days (~1.2 weeks)**.
   - `gpt-4o-mini` performed naive lexical addition: $2 + 1 = 3\text{ weeks}$.
2. **Question for Claude**: Should memory assembly pre-compute and inject normalized absolute temporal intervals (e.g. `[Calculated Interval: 9 days (~1.2 weeks)]`) when multiple temporal events are retrieved together, and how can this be done deterministically without online LLM calls?

---

## 5. Case Study D: PersonaMem Distractor Resistance (JEV Win vs. Anchor Loss)

### Metadata
- **Question ID**: `d5df0faf-8271-4b11-847a-c634d028139e`
- **Dataset**: `PersonaMem-32K` (Split: `dev_cohort`)
- **User Query**: Multiple-choice inquiry regarding user's active workout routine preference after an update.
- **Scores**:
  - AML Baseline 1: `0.0`
  - **RelayRouter JEV**: `1.0` (Won)
  - Anchor v2.6.1: `0.0` (Chose distractor option)

### Diagnostic Analysis
- In PersonaMem, 4 multiple-choice options are presented. Three are subtle distractors reflecting past preferences or adjacent hobbies mentioned by other conversation partners.
- **Anchor's Vulnerability**: Dense vector retrieval + BM25 ranked the general fitness conversation highest, but failed to separate the *superseded* routine from the *currently active* routine because the semantic similarity to the query keywords was nearly identical.
- **JEV's Strength**: The cross-encoder scored the full candidate hypothesis against the options, correctly identifying the recent update.
- **Latency Cost**: JEV took **642ms** on this query (unacceptable in AML production).
- **Question for Claude**: How can we design a candidate pruning strategy that runs JEV re-ranking *only* on the top-12 candidates for multiple-choice preference questions, ensuring latency stays well under 50ms?

---

## 6. Case Study E: Long-Horizon Synthesis (Anchor Win vs. JEV Loss)

### Metadata
- **Question ID**: `1:summarization:0`
- **Dataset**: `BEAM-100K` (Split: `full_1017`)
- **User Query**: Comprehensive cross-session project development timeline synthesis.
- **Scores**:
  - Baseline 1: `0.20`
  - RelayRouter JEV: `0.20`
  - **Anchor v2.6.1**: **`0.40`** (Won)

### Diagnostic Analysis
- In ultra-long contexts (100K tokens across multiple sessions), answering requires assembling evidence blocks from months apart.
- **Why Anchor Won**: Anchor's Tri-Tier Quotas (55% Events, 25% Episodes, 20% Raw) guaranteed that every key milestone had both an episode summary and an event anchor admitted into the budget.
- **Why JEV Failed**: JEV's pairwise cross-encoder prioritized a single densely-described milestone and pushed subsequent milestone candidates down the ranking, leading to lower timeline coverage.
- **Question for Claude**: How can a verification system preserve diversity and temporal spread across multi-session milestones without allowing a single dense cluster to dominate the top ranks?

---

## 7. How to Use the Companion Data Files

Researchers reviewing this project can inspect:
1. **Full 248 Traces JSONL**: [`benchmarks/evaluation_results/benchmark_248_full_traces.jsonl`](../../benchmarks/evaluation_results/benchmark_248_full_traces.jsonl)
   - Every line contains `id`, `dataset`, `query`, `baseline_score`, `jev_score`, `anchor_score`, `anchor_generated_answer`, and `top_retrieved_evidence`.
2. **Aggregated Benchmark Summary**: [`benchmarks/evaluation_results/summary.json`](../../benchmarks/evaluation_results/summary.json)
3. **Extraction Pipeline Examples**: [`docs/analysis_and_badcases/EXTRACTION_EXAMPLES.md`](EXTRACTION_EXAMPLES.md)
