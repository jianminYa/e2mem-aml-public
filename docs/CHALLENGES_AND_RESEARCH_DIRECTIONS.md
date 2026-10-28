# Strategic Research Brief & Open Challenges for External Consultation

> **Document Version**: 2.6.1-Release  
> **Evaluation Framework**: Agent Memory Leaderboard (AML 2026)  
> **Target Audience**: AI Architecture Researchers & Strategic Reviewers (Claude / External Review)  
> **Core Objective**: Systematic diagnosis of memory bottlenecks, latency-bounded verification (JEV), binary rubric optimization, and non-hallucinatory multi-hop expansion.

---

## 1. Executive Context & Benchmark Baseline

**E²-Mem Evidence Anchor (v2.6.1)** is currently the top-performing long-term memory engine on our 248-question balanced benchmark cohort across 5 diverse AML evaluation datasets:

| Benchmark Dataset | Test Cohort ($N$) | Metric Type | AML Baseline 1 | Anchor v2.4 | RelayRouter JEV | **Anchor v2.6.1 (Current SOTA)** | Net Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LoCoMo** | 32 | Multi-session Dialogue QA | 0.6250 | 0.6250 | 0.6562 | **0.6562** | **+0.0312** |
| **PersonaMem-32K** | 56 | Persona / Preference QA | 0.5179 | 0.5536 | **0.6250** | **0.5893** | **+0.0714** |
| **BEAM-100K** | 80 | Massive Context Synthesis | 0.3929 | 0.4188 | 0.4207 | **0.4467** | **+0.0538** |
| **LongMemEval-S** | 48 | Long Conversation Extraction | 0.5625 | 0.5625 | 0.5833 | **0.5625** | 0.0000 |
| **CLBench** | 32 | Multi-Constraint Regulation | 0.0625 | 0.0938 | 0.0312 | **0.0938** | **+0.0312** |
| **5-Dataset Macro** | **248** | **Unweighted Average** | **0.4322** | **0.4507** | **0.4633** | **0.4697** | **+0.0375** |

In full-scale evaluation across 897 test questions, v2.6 achieved **0.4394 Macro** (+0.0303 over Baseline 1, 101 Wins / 62 Losses).

### Hard Production Constraints
1. **Downstream Reader**: Fixed frozen `gpt-4o-mini-2024-07-18`. No prompt tuning of the reader or judge is permitted by AML rules; only the memory context payload returned by `POST /search` can be controlled.
2. **Search Latency SLA**: P95 latency must remain strictly **under 200ms**. Any solution requiring heavy multi-step LLM calls or high-latency cross-encoders during the search path will violate the production contract.
3. **Budget Trade-Offs**: While AML provides up to 117,760 tokens of input space, empirical testing confirms that unconstrained dialogue expansion dilutes `gpt-4o-mini` attention, degrading LoCoMo and PersonaMem. We maintain an 8,000-token discipline for dialogues, while dynamically allocating 24,000 tokens for document artifacts.
4. **Contract Guarantee**: Returned items must strictly satisfy $\text{len}(\text{results}) \le \text{top\_k}$.

---

## 2. Core Research Challenges & Strategic Consultation

We solicit concrete architectural, algorithmic, and data-engineering recommendations on the following four critical challenges.

---

### Challenge 1: The CLBench Paradox — Binary All-or-Nothing Scoring Under Complete Context

#### Empirical Observation
In our full diagnostic failure audit of all 32 CLBench test questions in the v2.6 benchmark:
- **68.8% of cases (22/32 questions)**: The complete statutory/guideline document (~5,695 tokens on average) was **100% present in the prompt context** passed to `gpt-4o-mini`. There was zero retrieval truncation.
- **Outcome**: The model scored **0.0** on all 22 questions because CLBench uses strict binary **All-or-Nothing checklist rubrics** ($7 \sim 15$ mandatory criteria per question). Missing even a single minor condition, reference citation, or formatting constraint produces a score of 0.
- **12.5% of cases (4/32 questions)**: Failed due to upstream API gateway timeouts or JSON formatting truncation.
- **9.4% of cases (3/32 questions)**: Evaluator variance where identical model responses scored 1 in one run and 0 in another.
- **9.4% of cases (3/32 questions)**: Fully passed all constraints, scoring 1.0.

#### The Core Dilemma
Standard RAG techniques assume failures are caused by missing chunks. Here, **the entire document is already in context**. Chunk-level re-ranking or section extraction actually risks stripping out necessary exceptions or cross-referenced clauses.

#### Consultation Questions for Claude
1. **Structural Pre-Processing & Layout Re-organization**:  
   Without omitting raw legal text, how can long-form statutory documents be re-indexed or reformatted at memory storage/assembly time so that `gpt-4o-mini`'s attention is guided directly to nested exceptions, definitions, and prerequisite constraints?
2. **Context Framing vs. Anti-Prompting Penalties**:  
   Can memory assembly inject neutral, structured metadata (e.g., Table of Contents, Section Hierarchy Trees, or explicit Clause-Exception mappings) without triggering LLM reader prompt drift or evaluation benchmark guardrail penalties?
3. **Selective Extraction vs. Monolithic Injection**:  
   If full injection suffers from attention fatigue on 15-constraint rubrics, is there a deterministic, zero-LLM method to dynamically cluster related statutory rules while guaranteeing 100% recall of associated exceptions?

---

### Challenge 2: Latency-Bounded Joint Evidence Verification (JEV)

#### Empirical Observation
In historical evaluations with RelayRouter JEV (Joint Evidence Verification):
- **Positive Impact**: JEV achieved **0.6250** on PersonaMem-32K (vs. Anchor's **0.5893**, representing a +3.57% net improvement and +2 resolved questions). JEV successfully filtered out spurious lexical matches in fine-grained user profiles.
- **Severe Negative Trade-offs**:
  1. **Latency Explosion**: Running cross-encoder verification across an unconstrained candidate pool ($M > 40$) escalated search latency to $450 \sim 850\text{ ms}$, severely breaching the 200ms P95 latency ceiling.
  2. **Domain Regression**: On CLBench, JEV dropped score from 0.0625 to 0.0312, and on BEAM-100K it dropped from 0.4467 to 0.4207, because generic cross-encoders penalized multi-session structural context.

#### The Core Dilemma
We want the +3.57% factual precision boost of JEV on entity/persona queries, but we cannot tolerate $> 200\text{ ms}$ search latency, nor regressions on document/synthesis tasks.

#### Consultation Questions for Claude
1. **Sub-Candidate Window Allocation**:  
   If JEV is restricted strictly to a top candidate slice (e.g., re-ranking only positions $k \in [1, 16]$ or $k \in [1, 24]$), what scoring fusion formulation ($\alpha \cdot \text{Anchor} + (1-\alpha) \cdot \text{JEV}$) prevents out-of-domain degradation?
2. **Conditional Gating**:  
   How can we design a zero-latency heuristic gate that triggers JEV **only** when dense-lexical retrieval confidence margin $\Delta = s_1 - s_2$ is narrow, or when the query is classified as a fine-grained attribute query (PersonaMem-type), while completely bypassing long documents (CLBench) and massive narrative synthesis (BEAM)?
3. **Ultra-Lightweight Scoring Alternatives**:  
   Are there viable sub-20ms alternatives to full cross-encoders (e.g., ColBERT-style token-level late interaction, BM25-IDF masked term coverage vectors, or in-memory quantized bi-encoder residual dot-products) that deliver the verification power of JEV within our strict latency budget?

---

### Challenge 3: Temporal State Updates & Downstream Recency Bias

#### Empirical Observation
In datasets like LoCoMo and LongMemEval-S, users frequently change states (e.g., moving from Seattle to San Francisco, canceling a marathon, changing dietary habits).
- Our **Intra-Channel PairGuard** identifies contradictory or updated propositions and pairs them atomically:
  $$\text{PairGuard}(e_{\text{current}}) \implies \{ e_{\text{current}}, e_{\text{superseded}} \}$$
- **Failure Mode**: When both statements are presented in the retrieved context with explicit timestamps (e.g., `[2026-03-01]` vs `[2026-05-15]`), `gpt-4o-mini` still frequently exhibits **order bias** (either recency bias or primacy bias), sometimes declaring the superseded fact as true or stating "the user gave conflicting statements."

#### Consultation Questions for Claude
1. **Context Ordering**:  
   Should contradictory/updated fact pairs be ordered chronologically (oldest $\to$ newest), reverse-chronologically (newest $\to$ oldest), or clustered with the newest fact as the anchor and the old fact explicitly referenced as historical background?
2. **Zero-Hallucination Temporal Annotations**:  
   How should temporal intervals and status transitions be rendered into text so that a frozen downstream LLM reader reliably resolves state progression without hallucinating assumptions?

---

### Challenge 4: Zero-LLM Multi-Hop Reasoning Without Lexical Drift

#### Empirical Observation
In our experimental v2.7 cycle, we attempted a 1-hop lexical entity expansion:
- When Top-8 Events were retrieved, entities were lexically extracted and queried via SQL `content LIKE ?`.
- **Result**: The macro score plummeted from **0.4697 to 0.4393** (-0.0304 across 4 datasets).
- **Root Cause**: In conversational datasets (LoCoMo, PersonaMem), character names (e.g., "Sam", "Evan") and common activity verbs appear in almost every session. Lexical expansion pulled in completely unrelated conversations (e.g., a GPS sensor conversation for a marathon query), flooding the 8K context and causing hallucinated answers.

#### The Core Dilemma
Multi-hop questions (e.g., connecting a friend's recommendation in Session 2 to an action in Session 10) genuinely require cross-session evidence bridging. However, online LLM graph extraction during search is prohibited by latency, while unconstrained lexical expansion causes fatal query drift.

#### Consultation Questions for Claude
1. **Entity Salience & Co-occurrence Constraints**:  
   How can a zero-LLM retrieval pipeline extract **truly salient bridge entities** (filtering out ubiquitous conversational anchors like speaker names and generic nouns) and apply bipartite or co-occurrence constraints to prevent topic drift?
2. **Graph Storage at Ingestion Time**:  
   Since ingestion (`POST /add`) has a more relaxed latency budget than search, what lightweight graph structure (e.g., subject-predicate-object triples, temporal anchor links, session co-reference chains) can be constructed during ingestion to enable fast, single-lookup deterministic traversal at query time?

---

## 3. Summary of Target Questions for Reviewers

When reviewing this codebase and architecture, please focus recommendations on:

1. **How to break the CLBench 0.0938 plateau** when the full document text is already in context under strict binary rubrics.
2. **How to architect a latency-bounded (< 50ms overhead, P95 < 200ms overall) residual JEV verification layer** that captures the +3.57% PersonaMem gain without damaging CLBench or BEAM.
3. **How to optimally format and order atomic temporal pairs** to guide `gpt-4o-mini` toward accurate state progression resolution.
4. **How to execute multi-hop relational bridging** using zero online LLM calls while completely avoiding lexical entity drift.
