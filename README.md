# **Queeree Pipeline Flow**

This document outlines the data flows and logic of the pipeline. It roughly comprehensively-covers the high-level orchestration flow, followed by detailed breakdowns of each internal component. Some parts are simplified down a bit, but considering the complexity of the current codebase, honestly, I think this is comprehensive enough for most purposes.

The design rationale of this pipeline is to be able to be as llm-agnostic as possible while leveraging as much of the available retrieval tooling to assist and augment the capability of whatever model is loaded up. Preferably, we're using a reasoning-trained model for the LLM, but any VLM is compatible with this pipeline and can be swapped out relatively easily within the codebase to support an ensemble of models to handle the two modalities. (e.g., using a VLM for MLLM task, whilst the reasoning and textual ones are handled by a text-only LLM)

Crucially, we implemented this under the assumption that it'd prove more beneficial to, instead of tuning a model for classification of DFK tasks, instead emulate a content moderation agent through automated queries. We posit that the inductive bias of the DFK task is _not_ for the "detection", rather it would be more on the "reasoning" and "verification" of facts in the matter at hand, thus the training of the LLM may prove counterproductive to the core objective, that is verifying, and through that, _identifying_ cases of DFK. Note that tuning within this framework _can_ still be done, most particularly in the embedding models used, but all in all, again, it was designed to be as model-agnostic as possible.

---

## 1. General System Overview

The orchestrator (`app/pipeline/orchestrator.py` & `app/main.py`) acts as the entry point for requests and coordinates various specialized modules (extraction, classification, fact-checking, and legal analysis). The progress is then streamed to the client via Server-Sent Events (SSE).

```mermaid
graph TD
    A[Client Request: Text + Optional Image] --> B{Image Present?}
    B -- Yes --> C[Image Context Extractor LLM]
    C --> D[Append Extracted Context to Text]
    B -- No --> D

    D --> E[Classifier Module]
    E --> F{Categories & Flags Found?}
    F -- No --> G[Pipeline End: Safe]

    F -- Yes --> H{Needs Fact\nVerification?}
    H -- Yes --> I[Fact Checker Module]
    I --> J{Fact Check Status}

    J -- "FALSE (Debunked)" --> K[Intention Checker LLM]
    K --> L[Add 'Disinformasi' or 'Hoaks' Category]

    J -- "UNVERIFIED" --> M[Add 'Misinformasi' Category]

    J -- "TRUE (Verified)" --> N[Keep Original Categories]

    H -- No --> N
    L -.-> N
    M -.-> N

    N --> O{Any Categories Left?}
    O -- No --> P[Pipeline End: Safe]
    O -- Yes --> Q[Law Retriever Module]

    Q --> R{Laws Found?}
    R -- Yes --> S[Law Analyzer Module]
    R -- No --> T[Final Summary Generator]

    S --> T
    T --> U[Final Response: Flagged + Justification + Laws]
```

---

## 2. Classifier Flow

The classifier (`app/pipeline/classifier.py`) runs parallel LLM calls to determine categorical violations. It resolves disagreement by taking a simple majority vote.

```mermaid
graph TD
    A[Input Content + Image Data] --> B[Generate N Parallel LLM Requests]
    B --> C[LLM Vote 1]
    B --> D[LLM Vote 2]
    B --> E[LLM Vote N]

    C --> F[Extract Categories & needs_verification]
    D --> F
    E --> F

    F --> G[Aggregate Votes]
    G --> H{Vote Count >= Threshold \n N / 2 + 1 ?}

    H -- Yes --> I[Keep Category / Flag]
    H -- No --> J[Discard Category / Flag]

    I --> K[Filter against allowed categories]
    K --> L[Return Final Categories & Verification Flag]
```

---

## 3. Fact-Checker Flow

When the classifier flags claims as needing verification, the fact checker (`app/pipeline/fact_checker.py`) executes three independent parallel paths. It utilizes a web scraper (`app/pipeline/retrieval.py`) to gather real-world evidence and acts iteratively if the gathered evidence is insufficient.

```mermaid
graph TD
    A[Input Claim] --> B[Execute 3 Parallel Verification Paths]

    %% Path 1
    B --> P1[Path 1: Standard Search]
    P1 --> P1_Q[Generate Standard Query]
    P1_Q --> P1_S[Search Web]
    P1_S --> P1_E["LLM Eval: Is Evidence Sufficient?"]
    P1_E --> P1_C{Sufficient?}
    P1_C -- No (loop < max) --> P1_R[Refine Query]
    P1_R --> P1_S
    P1_C -- Yes / Max Loops --> P1_Res[Output Status + Reasoning]

    %% Path 2
    B --> P2[Path 2: Contrary/Debunk Search]
    P2 --> P2_Q[Generate Contrary Query]
    P2_Q --> P2_S[Search Web]
    P2_S --> P2_E[LLM Eval: Is Evidence Sufficient?]
    P2_E --> P2_C{Sufficient?}
    P2_C -- No (loop < max) --> P2_R[Refine Query]
    P2_R --> P2_S
    P2_C -- Yes / Max Loops --> P2_Res[Output Status + Reasoning]

    %% Path 3
    B --> P3[Path 3: Pure LLM Reasoning]
    P3 --> P3_L[LLM Extrapolates Likelihood N times]
    P3_L --> P3_Res[Output Status + Consistency Score]

    P1_Res --> C[Decision Aggregation]
    P2_Res --> C
    P3_Res --> C

    C --> D{'Evaluate Priority'}
    D -- 'Standard Sufficient' --> E[Use Standard Result]
    D -- 'Contrary Sufficient' --> F[Use Contrary Result]
    D -- 'Neither, but Reasoning Confident' --> G[Use Reasoning Result]
    D -- 'Ambiguous / Conflicts' --> H[Fallback: UNVERIFIED]

    E --> I[Return Fact Check Summary, Deduplicated Sources, and Final Status]
    F --> I
    G --> I
    H --> I

    style P1 stroke:#4CAF50,stroke-width:2px;
    style P2 stroke:#F44336,stroke-width:2px;
    style P3 stroke:#2196F3,stroke-width:2px;
```

---

## 4. Intention Checker Flow

Used exclusively if a claim is explicitly debunked (FALSE). The Intention Checker (`app/pipeline/intention_checker.py`) evaluates _why_ the false claim exists, distinguishing between coordinated malice ('Disinformasi') and casual misinformation/rumors ('Hoaks').

```mermaid
graph TD
    A[Debunked Content + Fact-Checker Reasoning] --> B[LLM Prompt: Intention Analysis]
    B --> C[LLM deduces context and malicious intent]
    C --> D{Is Intent clearly malicious?}
    D -- Yes --> E[Category: Disinformasi]
    D -- No / Unclear --> F[Category: Hoaks]
    E --> G[Return Sub-Category & Reasoning]
    F --> G
```

---

## 5. Law Retriever Flow

To ground moderation decisions, relevant Indonesian laws are searched locally (`app/pipeline/law_retriever.py`) and summarized.

```mermaid
graph TD
    A[Flagged Categories + Content Snippet] --> B[LLM Generates Search Query]
    B --> C[Retrieve from Local HNSWLib Vector Index]
    C --> D[Embed Query via SentenceTransformer]
    D --> E[Quantize to Binary]
    E --> F[Prefilter Top N Passages via KNN]
    F --> G[Hierarchical Re-ranking: \n Passage + Title Similarity]
    G --> H[Top K Candidate Laws]
    H --> I[Format Raw Hits as Context]
    I --> J[LLM Summarizes relevance to user content]
    J --> K[Return Markdown Summary & Specific Article Strings]
```

---

## 6. Law Analyzer Flow

When specific laws are found, the system performs a localized sentence-by-sentence analysis (`app/pipeline/law_analyzer.py`) to pinpoint exactly _which parts of the user content_ violated the law, and aggregates reasoning consensuses.

```mermaid
graph TD
    A[Content + Retrieved Articles + Fact Check Context] --> B[Parallel Analysis per Law Pasal]
    B --> C[Run N LLM instances for each Pasal]

    C --> D{Does Content Violate Pasal?}
    D -- Majority Yes --> E[Extract Violating Segments & Explanations]
    D -- Majority No --> F[Discard Pasal/Article]

    E --> G[Resolve Segment Offsets & Intersections]
    G --> H[Group segments overlapping with same substring matches]

    H --> I[Execute Reason Clustering LLM]
    I --> J[Generate Final Cohesive Multi-run Reason LLM]

    J --> K[Return specific violated segments, exact string highlight, and consolidated reason per Pasal]
```
