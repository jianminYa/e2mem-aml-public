# Empirical Retrieval Traces & Bad Case Deep-Dive

> **Document**: Benchmark Diagnostic Failure Analysis  
> **Repository**: E²-Mem Evidence Anchor (v2.6.1)  
> **Harness**: Official AML Benchmark (`gpt-4o-mini` reader, official rubrics)  

This document presents actual bad cases from our 248-question balanced benchmark. For each case, we show the **User Query**, the **Retrieved Memory Payload**, the **Downstream Generated Answer**, the **Evaluation Score**, and the **Root Cause Diagnosis**.

---

## Case 1: LoCoMo Temporal Math & Relative Date Resolution

- **Question ID**: `conv-41:63` (Dataset: LoCoMo)
- **User Query**: `"How many weeks passed between Maria adopting Coco and Shadow?"`
- **Official Score**: `0.0` (Failure)

### Retrieved Memory Payload (Top Hits)
```text
[Hit 1 | Score: 1.184]
[Event Date: 2023-07-28]
Event: Maria got a new puppy named Coco two weeks ago.
Episode: In a recent conversation, Maria excitedly told John about her new puppy, Coco, whom she got two weeks ago.
Source: I got a puppy two weeks ago! Her name's Coco.

[Hit 2 | Score: 0.901]
[Event Date: 2023-08-06]
Event: Maria adopted a puppy named Shadow from a shelter.
Episode: Maria talked about adopting a new puppy named Shadow from a shelter last week.
Source: I just adopted this cute pup from a shelter last week.

[Hit 3 | Score: 0.883]
[Conversation Time: 2023-08-11T00:10:00+00:00 | Speaker: assistant]
Maria: Hey John! Long time no talk! Guess what - I got a puppy two weeks ago! Her name's Coco and she's adorable.

[Hit 4 | Score: 0.733]
[Conversation Time: 2023-08-13T15:14:00+00:00 | Speaker: assistant]
Maria: Her name is Shadow! She's full of energy and always puts a smile on my face.
```

### Downstream Generated Answer (`gpt-4o-mini`)
> *"Three weeks passed between Maria adopting Coco and Shadow."*

### Root Cause Diagnosis
1. **Retrieval Precision was 100%**: The memory engine successfully retrieved both events and raw dialogue turns at Rank 1, 2, 3, and 4.
2. **Reader Arithmetic Confusion**:
   - Hit 3 occurred on August 11, 2023 ("got puppy two weeks ago" $\to$ July 28, 2023).
   - Hit 4 occurred on August 13, 2023 ("adopted last week" $\to$ August 6, 2023).
   - The actual elapsed time between July 28 and August 6 is **9 days (~1.2 weeks)**.
   - `gpt-4o-mini` mechanically added $2\text{ weeks} + 1\text{ week} = 3\text{ weeks}$, misunderstanding that both utterances were made in the same week of August.
3. **Key Takeaway for Memory Architecture**: Can memory retrieval normalize relative temporal expressions into **absolute date intervals** (e.g. `[Adoption Period: 2023-07-28 to 2023-08-06 (Interval: ~9 days)]`) before feeding context to the reader?

---

## Case 2: CLBench Complete Context with Binary Rubric Failure

- **Question ID**: `994a24cd-3fae-4ad5-9d1e-1f26b25113e4` (Dataset: CLBench)
- **User Query**: `"Yes but I mean can't you tell me which personality disorders are associated with this domain so I can have an idea what the child might be diagnosed with?"`
- **Official Score**: `0.0` (Binary Failure)

### Retrieved Memory Payload
- **Hit 1**: Raw document chunk of length **5,481 characters** containing the entire Clinical Diagnostic Assessment Manual and Pediatric Boundary Guideline.
- **Context Integrity**: Zero retrieval loss; 100% of the statutory rules, exceptions, and disclaimers were present in context.

### Downstream Generated Answer (`gpt-4o-mini`)
> *"I must remind you to de-identify any data and avoid sharing personally identifiable information (PII) before we proceed.*  
> *To assist you effectively, I need the following information:*  
> *1. Assessment Details: Exact name and version of the assessment used.*  
> *2. Reporter Type: Specify whether completed by the child or by a parent/clinician.*  
> *3. Full Assessment Results: I need at least one complete set of assessment results.*  
> *Once I have this information, I can help you analyze the results... Please note that I cannot provide diagnoses or treatment plans."*

### Root Cause Diagnosis
1. **The Compliance vs. Answering Conflict**: The prompt asked what disorders might be associated with a domain. The context contained strict clinical guidelines stating that personality disorders cannot be formally diagnosed in children under 18 without longitudinal assessment and parental reporting.
2. `gpt-4o-mini` adopted the defensive posture and refused to list the associated disorders without intake data.
3. The evaluation rubric contained a strict 12-item checklist requiring explicit mention of borderline, schizotypal, and avoidant traits. Because the model refused, it satisfied 0 items on the checklist, scoring **binary 0.0**.
4. **Key Takeaway for Memory Architecture**: In regulatory and legal tasks, how can memory assembly present guidelines so that the model understands that answering a hypothetical query does not violate the operational rules?

---

## Case 3: PersonaMem Budget Plan Update Confusion

- **Question ID**: `11102cd9-63fe-4eed-9e76-3416c176bcfa` (Dataset: PersonaMem-32K)
- **User Query**: `"I recently updated the budget plan I had shared with friends."`
- **Official Score**: `0.0` (Multiple-choice failure)

### Retrieved Memory Payload (Top Hits)
```text
[Hit 1 | Score: 1.323]
[Event Date: 2023-10-01]
Event: Alex shared their budget with close friends for feedback.
Episode: Alex has been actively enhancing their financial management skills...

[Hit 2 | Score: 1.097]
[Event Date: 2023-10-01]
Event: Alex created a comprehensive monthly budget outlining income, expenses, and savings objectives.

[Hit 3 | Score: 0.971]
[Speaker: user]
User: I've been working diligently to enhance my financial management skills lately, as I've come to realize how crucial effective budgeting is...
```

### Downstream Generated Answer (`gpt-4o-mini`)
> *(b) It's a great initiative to update the budget plan and share it. It's important to get feedback from friends. How are they responding to it? <final_answer>*

### Root Cause Diagnosis
1. In PersonaMem, the ground truth was option (d): Alex had previously shared a preliminary budget, but had *not* yet shared the revised version, or friends had already responded to the previous version with specific comments.
2. Hit 1 mentioned sharing with friends, but lacked the subsequent turn where Alex explained that the update was kept private.
3. The PairGuard module did not identify the second turn because the wording did not trigger explicit polarity reversal words ("canceled", "not", "refused").
4. **Key Takeaway**: Implicit state modifications (updating a document privately vs sharing an older version) require semantic transition detection beyond simple lexical antonyms.

---

## Case 4: LoCoMo Multi-Hop Query Drift (v2.7 Regression Analysis)

- **Question ID**: `conv-49:27` (Dataset: LoCoMo)
- **User Query**: `"What electronic device could Evan gift Sam to help him keep up with his fitness goals?"`
- **Ground Truth**: Fitness tracker / smartwatch.

### v2.7 Lexical 1-Hop Expansion Failure
During the experimental v2.7 run, naive 1-hop expansion pulled records based on `content LIKE '%Evan%'` and `content LIKE '%Sam%'`.
- **Injected Bridge Candidates**:
  1. *"Sam appreciates Evan's encouragement."*
  2. *"Evan provided support and encouragement to Sam."*
  3. *"Evan helped Sam configure a GPS sensor on his key chain."* (from an unrelated trip session)
- **Resulting Answer**: The model stated Evan should gift a *"GPS key chain sensor"*, abandoning the correct *"fitness tracker"*.
- **Score**: Dropped from `1.0` to `0.0`.
- **Key Takeaway**: Unconstrained lexical entity matching pulls in universal character interactions, creating query drift. Multi-hop bridging must require **query-predicate relevance constraints**.
