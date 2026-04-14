---
slug: embeddings
sources:
- hav4ik.github.io
tags: []
title: Embeddings and Representation Learning
updated: '2026-04-14'
---

# Embeddings and Representation Learning

**Embeddings** are learned vector representations of entities (e.g., words, queries, documents, images, users, items) such that geometric relationships in vector space (e.g., cosine similarity or Euclidean distance) correspond to semantic or functional similarity. This is a core technique in modern machine learning for **representation learning** and underpins many retrieval and ranking systems.

This page focuses on embeddings in the context of **information retrieval and web search**, where embeddings are central to *embedding-based retrieval* and later stages of *learning-to-rank*.

See also: [[learning-to-rank]], [[information-retrieval]], [[search]], [[recommender-systems]], [[bert]], [[contrastive-learning]], [[approximate-nearest-neighbors]].

---

## Why embeddings matter in search and ranking systems

Modern search engines can be abstracted into stages:

- **Indexing (offline, continuous):** extract and store searchable signals/features for documents.
- **Retrieval / matching (online, per query):** retrieve a candidate set of documents (top-*k*).
- **Ranking / re-ranking (online):** sort candidates by predicted relevance (and possibly personalization).

Embeddings contribute primarily to:

- **Vector indexing:** storing embeddings for documents (and sometimes queries) in a dedicated vector index.
- **Embedding-based retrieval:** retrieving candidates via nearest-neighbor search in embedding space.
- **Feature generation for rankers:** embeddings (often compressed) are used as part of large feature sets for learning-to-rank models.

This connects embeddings to both:
- **First-stage retrieval** (high recall, fast)
- **Later-stage ranking** (higher precision, uses many signals)

---

## Embeddings in search engine indexing

In addition to classical lexical indexes, modern search engines may maintain multiple kinds of indexes:

- **Inverted index (posting list):**
  - Maps terms → documents.
  - Supports term-based scoring like **TF-IDF** and **BM25**.
- **Vector index:**
  - Stores **embeddings** of documents and queries.
  - Embeddings are “usually computed using state-of-the-art contrastive learning neural networks.”
  - Examples mentioned:
    - Visual search: **SigLIP** embeddings
    - Text search: **BERT-like** embeddings
- **Feature index:**
  - Built from “thousands of different signals” and can include **compressed embeddings**.
  - Used for re-ranking.

Cross-references: [[bm25]], [[tf-idf]], [[vector-database]], [[contrastive-learning]].

---

## Embedding-based retrieval (dense retrieval)

### Core idea
Given a query, compute its embedding vector and retrieve the **k nearest document embeddings** from the vector index using a similarity measure such as:

- **Cosine similarity**
- **Euclidean distance**

This is commonly part of a **hybrid retrieval** approach at web scale:

- **Keyword/entity matching** via inverted index (lexical retrieval)
- **Embedding-based retrieval** via vector index (semantic retrieval)

Additional retrieval enhancements mentioned:

- **Knowledge graph query expansion** can be used to broaden/expand the query and retrieve more relevant documents.

Cross-references: [[dense-retrieval]], [[hybrid-retrieval]], [[knowledge-graphs]].

### Approximate nearest neighbor (ANN) for scalability
The source highlights a systems/scaling point:

- Metric trees (e.g., k-d trees) are typically **not used** in large-scale search engines due to:
  - “slow \(O(\log n)\) complexity” and
  - “large memory consumption.”
- Instead, **Approximate Nearest Neighbors (ANN)** methods (e.g., “LHS or PCA hashing”) are used to achieve “close to \(O(1)\)” retrieval complexity.

Cross-references: [[approximate-nearest-neighbors]], [[nearest-neighbor-search]], [[vector-index]].

**Note/possible contradiction to flag:**  
The claim that k-d trees are avoided because \(O(\log n)\) is “slow” and that ANN is “close to \(O(1)\)” is a high-level engineering statement. In practice, ANN methods typically provide *sublinear* expected query time with strong constant-factor tradeoffs and hardware/system optimizations; strict \(O(1)\) is not generally guaranteed. This page preserves the source phrasing but notes that complexity is often workload- and implementation-dependent.

---

## Representation learning signals used for ranking

Embeddings rarely act alone in production ranking. The source emphasizes:

- **Feature engineering remains extremely important**: “the more expressive your features are, the better your ranking layer will perform.”
- Learned representations (embeddings) are often combined with:
  - lexical match features (BM25, term overlap),
  - link analysis signals (e.g., PageRank),
  - behavioral signals (e.g., clicks),
  - metadata/content quality signals,
  - and many other domain-specific features.

Cross-references: [[feature-engineering]], [[pagerank]].

---

## How embeddings connect to Learning to Rank (LTR)

After retrieval produces a candidate set, **Learning to Rank (LTR)** models re-order results.

The source frames LTR as learning a function \(f(\mathbf{q}, \mathcal{D})\) that produces a ranking (often via scoring each document and sorting by score). Embeddings contribute to LTR by:

- improving candidate set quality (better retrieval recall/semantic matching),
- providing learned semantic features for the ranker,
- enabling multimodal ranking (e.g., combining text and image embeddings).

Cross-reference: [[learning-to-rank]].

---

## Relevance signals and their relationship to representations

The source describes relevance as often measured using a combination of:

- **Human-labeled judgments**
  - expensive but high-quality; often multi-grade relevance (e.g., 1–5)
- **Click-through rate (CTR)**
  - cheap but biased (e.g., position bias)
- **Conversion rate**
  - task/business dependent (e.g., buys per search in e-commerce)

Embeddings/representation learning influences these signals indirectly by changing what gets retrieved and shown, which then changes observed clicks/conversions, and by supporting personalization or query understanding.

Cross-references: [[click-models]], [[counterfactual-learning]], [[unbiased-learning-to-rank]].

---

## Evaluation metrics (embedding-aware pipelines)

Even though embeddings are not a metric by themselves, embedding-based retrieval and representation learning are typically evaluated end-to-end with ranking metrics, including:

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **ERR** (Expected Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain)

### NDCG definition (as used in LTR)
Define:

\[
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
\]

and:

\[
NDCG@T = \frac{DCG@T}{\max DCG@T}
\]

where \(l_i\) is the relevance label at rank \(i\), and \(T\) is truncation level (e.g., 10).

### ERR definition (as used in LTR)
\[
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right), \quad \text{where}\; R_i = \frac{2^{l_i} - 1}{2^{l_m}}
\]

Cross-references: [[ndcg]], [[mrr]], [[map]], [[evaluation-metrics]].

---

## Embeddings vs. lexical retrieval: complementary strengths

From the source’s description of hybrid retrieval, a practical summary is:

- **Lexical retrieval (BM25/TF-IDF on inverted indexes)** is strong for:
  - exact term matches, rare entities, precise keyword queries
- **Embedding-based retrieval (vector search)** is strong for:
  - semantic similarity, paraphrase matching, conceptual matches, multilingual alignment (in some models)

In web-scale systems, **hybrids** are used to balance precision/recall and robustness.

Cross-references: [[bm25]], [[dense-retrieval]], [[hybrid-retrieval]].

---

## Model families used to produce embeddings (as referenced)

The source mentions embeddings are “usually computed using state-of-the-art contrastive learning neural networks,” with examples:

- **BERT-like** models for text embeddings
- **SigLIP** embeddings for vision / visual search

Cross-references: [[bert]], [[contrastive-learning]], [[multimodal-ml]].

---

## Practical notes and caveats

- **Embeddings are infrastructure-dependent in search:**
  - They require a **vector index** and an ANN retrieval layer.
- **Bias and feedback loops:**
  - Changes in embedding retrieval can change what users see, which changes clicks, which can reinforce bias unless corrected with methods like [[unbiased-learning-to-rank]].
- **Page design affects interaction biases:**
  - The source notes eye-tracking evidence that UI design changes can “flatten” attention distribution across ranks, affecting position bias estimators.

Cross-references: [[position-bias]], [[trust-bias]], [[selection-bias]].

---

## Related pages

- [[learning-to-rank]] (LTR objectives and ranking models)
- [[approximate-nearest-neighbors]] (ANN algorithms for vector search)
- [[contrastive-learning]] (common training paradigm for modern embedding models)
- [[bm25]] and [[tf-idf]] (lexical retrieval baselines used alongside embeddings)
- [[bert]] (text embedding model family)
- [[vector-database]] / [[vector-index]] (storage and serving layer for embeddings)
- [[counterfactual-learning]] and [[unbiased-learning-to-rank]] (learning from biased click signals)

---

## Source integration notes

- This page is initialized from a source focused on **Learning to Rank in Web Search**, integrating the parts relevant to embeddings:
  - vector indexes,
  - embedding-based retrieval,
  - ANN scaling considerations,
  - and how embeddings feed into ranking pipelines.
- **Potential contradiction/nuance recorded:**
  - The simplified complexity statements about k-d trees vs. ANN (“\(O(\log n)\)” vs “close to \(O(1)\)”) are directional and practical rather than strict worst-case guarantees.