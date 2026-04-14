---
slug: vector_search
sources:
- hav4ik.github.io
tags: []
title: Vector Search and ANN
updated: '2026-04-14'
---

# Vector Search and ANN

Vector search (also called embedding-based retrieval) is a family of retrieval techniques where both queries and documents/items are represented as dense numeric vectors (“embeddings”), and retrieval is performed by finding the nearest vectors according to a similarity function (commonly cosine similarity or Euclidean distance). In modern search stacks, vector search is typically used for *top‑k retrieval* (a.k.a. “matching” or “Level‑0 ranking”), and is often combined with keyword-based retrieval.

This page focuses on **vector search** and **Approximate Nearest Neighbor (ANN)** methods that make nearest-neighbor retrieval feasible at large scale, and how they fit into a broader search system and ranking pipeline (see also [[learning_to_rank]]).

---

## Where vector search fits in a modern search engine

A common high-level architecture for search systems is:

1. **Indexing (offline / continuous)**: extract signals and build indexes.
2. **Top‑k retrieval (online)**: quickly fetch a candidate set of potentially relevant documents.
3. **Ranking / re-ranking (online)**: sort candidates by relevance and/or personalization, often with ML-based Learning-to-Rank.

Vector search primarily powers step (2), but also produces features that can be used in ranking/re-ranking stages (step (3)).

### Index types commonly used together

Modern web-scale and recommender-style search engines may maintain multiple indexes, each optimized for different retrieval paradigms:

- **Inverted index (posting lists)**  
  - Maps terms → documents.
  - Enables classic lexical retrieval and scoring such as TF‑IDF and BM25.
  - Useful for exact term matching and entity matching.

- **Vector index**  
  - Stores dense embeddings for documents (and sometimes pre-computed query embeddings, depending on the system).
  - Enables nearest-neighbor retrieval by vector similarity (cosine or Euclidean).

- **Feature index**  
  - Stores a large collection of engineered signals and/or compressed neural embeddings.
  - Used heavily in later-stage ranking (re-ranking).

This “multi-index” view aligns with modern search stacks where retrieval is **hybrid**: lexical + embedding-based retrieval, sometimes augmented by query expansion via knowledge graphs.

Cross-references:
- Ranking and re-ranking with ML: [[learning_to_rank]]
- Classical lexical retrieval signals (e.g., BM25): [[bm25]] (if present)

---

## Embeddings for vector search

Vector search depends on embeddings learned by neural models, commonly via **contrastive learning** approaches.

Examples mentioned in the source material:

- **Visual search**: embeddings such as **SigLIP**-style representations.
- **Text search**: **BERT-like** embeddings.

Typical workflow:

- Compute document embeddings during indexing and store them in the **vector index**.
- Compute a query embedding at request time.
- Retrieve the **k nearest document vectors** by similarity.

---

## Similarity functions used for nearest neighbors

Common choices:

- **Cosine similarity**
- **Euclidean distance**

Implementation note: many ANN systems support multiple metrics or implement cosine similarity via normalized vectors and inner product.

---

## Why Approximate Nearest Neighbor (ANN) is used

Exact nearest neighbor search over large corpora can be too slow and memory-intensive. The source text highlights a key practical point for **large-scale** search engines:

- **Metric trees** (e.g., k‑d trees) are described as *not used* in large-scale search engines due to:
  - slow behavior at scale despite the nominal \(O(\log n)\) complexity, and
  - large memory consumption.

Instead, large systems use **Approximate Nearest Neighbor (ANN)** methods to achieve **close to \(O(1)\)** retrieval time in practice.

> Note: The “close to \(O(1)\)” claim reflects an engineering perspective and typical behavior of hashed/quantized ANN lookups, not a universal theoretical guarantee for all ANN methods and all distributions.

---

## ANN families and techniques (from the new source)

The source explicitly mentions ANN approaches such as:

- **Hashing-based ANN**
  - e.g., “LHS” (likely intended as LSH / Locality-Sensitive Hashing; the source text uses “LHS”)
  - **PCA hashing**

These approaches trade exactness for speed by mapping vectors into codes/buckets so that candidate neighbors can be found without scanning the entire dataset.

### Contradiction / ambiguity note
- The source states “ANN search (like **LHS** or **PCA hashing**)”. In common terminology, **LSH** (Locality-Sensitive Hashing) is the standard acronym; “LHS” is ambiguous and may be a typo or refer to a different method. This page preserves the source phrasing but flags the ambiguity.

---

## Hybrid retrieval: vectors + keywords

For web-scale search engines, retrieval often uses a **hybrid** of:

- **Keyword / entity matching** via the inverted index
- **Embedding-based retrieval** via the vector index

This improves recall (semantic matching) while retaining precision/constraints provided by lexical matching. Hybrid retrieval is especially useful when:
- queries are short/ambiguous,
- vocabulary mismatch exists (synonyms, paraphrases),
- users search with natural language.

---

## Relationship to ranking and Learning to Rank (LTR)

Vector search is typically used to fetch a candidate set (top‑k) which is then refined by a ranker. The new source emphasizes:

- **Retrieval** (“Level‑0 ranking” / “matching”) produces top‑k candidates.
- **Ranking** is what “actually makes search engines work” by ordering retrieved documents by relevance (and optionally user preferences).
- Large search engines rely on **Machine Learning / Learning-to-Rank** in later stages.

Cross-reference: [[learning_to_rank]]

---

## Biases and evaluation caveats (why retrieval ≠ relevance)

While not specific to ANN mechanics, the source discusses that using user interaction (e.g., clicks) as a proxy for relevance is biased due to effects like:

- position bias,
- selection bias,
- trust bias.

This matters to vector search systems because:
- retrieval quality is often evaluated and improved using interaction logs,
- hybrid retrieval and ANN changes can affect exposure distributions, thereby changing observed click behavior.

Cross-reference: [[learning_to_rank]] (especially unbiased / counterfactual LTR sections)

---

## Practical notes and engineering takeaways

- **Indexing is continuous and offline**: embeddings and features are computed ahead of time and stored in appropriate indexes.
- **Vector search uses nearest neighbors**: compute query embedding online, retrieve nearest doc embeddings from a vector index.
- **ANN is essential at scale**: tree-based exact/metric structures are typically avoided in web-scale settings; approximate methods are used for latency and memory reasons.
- **System design affects behavior**: UI and ranking presentation can change user behavior patterns (relevant if learning from clicks), so retrieval/ranking evaluation should account for these shifts.

---

## Open questions / areas to expand

This page currently reflects ANN mentions from the provided source (hashing and PCA hashing) and high-level search architecture context. Common ANN techniques not covered in the source (e.g., graph-based ANN, vector quantization/IVF-style indexes) are intentionally not asserted here to avoid adding unsourced specifics; they can be added if/when additional sources are provided.