---
slug: embeddings
sources:
- hav4ik.github.io
- blog.reachsumit.com
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

### Cascade / multi-stage ranking context (new)
A common production pattern (in both [[search]] and [[recommender-systems]]) is a **cascade ranking system**:

- **Recall / retrieval:** very fast, high-recall candidate generation (lexical, dense, or hybrid).
- **Pre-ranking:** lightweight model that filters and scores a *large* candidate set under tight latency.
- **Ranking / re-ranking:** heavier models that score a much smaller set with richer interactions and features.

Operational motivation highlighted by the new source:
- Systems may have **tens of millions** of candidates.
- Even small latency increases (e.g., +100ms) can measurably hurt user experience and revenue, pushing architectures toward multi-stage designs.

Cross-references: [[learning-to-rank]], [[information-retrieval]], [[recommender-systems]].

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

### Two-tower embedding storage and “freezing” (new)
In industrial retrieval and pre-ranking, embeddings are often produced by **two-tower (dual-encoder) architectures** (see [[dense-retrieval]] for the retrieval framing):

- A **query/user tower** produces an embedding for the request context.
- A **document/item/ad tower** produces an embedding for each candidate.
- The two vectors interact via a **late interaction** operation (often a dot product / inner product) to produce a similarity score.

A key serving optimization emphasized by the new source:
- The **document/item tower embeddings are often frozen after training** and stored in an indexing service (i.e., a [[vector-index]] / [[vector-database]]) for fast inference-time lookup.
- In some deployments, both towers may be effectively “frozen” for serving, with periodic offline retraining and index refresh (e.g., daily).

Cross-references: [[vector-index]], [[vector-database]], [[dense-retrieval]], [[nearest-neighbor-search]].

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

**Note/possible contradiction to flag (preserved):**  
The claim that k-d trees are avoided because \(O(\log n)\) is “slow” and that ANN is “close to \(O(1)\)” is a high-level engineering statement. In practice, ANN methods typically provide *sublinear* expected query time with strong constant-factor tradeoffs and hardware/system optimizations; strict \(O(1)\) is not generally guaranteed. This page preserves the source phrasing but notes that complexity is often workload- and implementation-dependent.

---

## Two-tower (dual-encoder / bi-encoder) architectures for embeddings (new)

Two-tower models are widely used across **search**, **ads**, and **recommendation** for retrieval and especially **pre-ranking** due to a strong accuracy/latency tradeoff.

### Basic structure
- **Inputs:** e.g., (query, document) or (user, item/ad).
- Each side passes through:
  - embedding layers (for sparse/categorical features, tokens, IDs),
  - optional DNN layers,
  - producing a **latent vector representation**.
- **Scoring:** similarity is computed via **inner product** (dot product) or related similarity in embedding space.

This is a **representation-based** (a.k.a. “embedding-based”) approach:
- Query and document embeddings are computed independently, then compared at the end (**late interaction**).
- Independence enables precomputing and indexing one side (usually documents/items).

Cross-references: [[dense-retrieval]], [[contrastive-learning]], [[vector-index]].

### Terminology: dual encoder variants
The new source distinguishes common dual-encoder forms:

- **Siamese Dual Encoder (SDE):**
  - Two identical subnetworks (shared parameters).
  - Historically proposed for similarity/distance tasks (e.g., signature verification).
- **Asymmetric Dual Encoder (ADE):**
  - Two distinct (non-shared) encoders.
  - Used when asymmetry between inputs is desired (e.g., query vs. document differences).

A further variant discussed:
- **ADE with Shared Projection Layer (ADE-SPL):** share (some) projection layer even if encoders differ.

Cross-references: [[representation-learning]] (if present), [[dense-retrieval]].

### Reported effectiveness claim (potentially conflicting/nuanced)
The new source summarizes results from a QA retrieval study:

- It reports **SDEs perform significantly better than ADEs**, attributing worse ADE performance to embedding the two inputs into **disjoint embedding spaces** (hurting retrieval quality).
- It also reports ADE performance can be improved to match or exceed SDE by **sharing a projection layer** (ADE-SPL), while sharing/freezing token embedders (ADE-STE, ADE-FTE) gives only marginal gains.

**Nuance / possible contradiction with common practice (explicitly noted):**
- This “SDE > ADE” conclusion is task- and implementation-dependent; many production dense retrieval systems do use asymmetric encoders successfully (e.g., different capacity, modalities, or feature sets per side). The key risk is *misaligned spaces*, which can be mitigated by shared components, joint training objectives, or alignment losses. This page records the source’s claim while noting it is not universally true across all retrieval settings.

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

### Pre-ranking vs ranking: interaction tradeoffs (new)
The new source frames an architectural tradeoff:

- **Pre-ranking** favors **fast** models because it scores many candidates.
  - Two-tower models are common here because they keep query-document interaction minimal (late interaction).
- **Ranking / re-ranking** can use more expensive **interaction-rich** models because it scores fewer candidates.

This aligns with the broader cascade idea discussed earlier.

Cross-references: [[learning-to-rank]].

---

## How embeddings connect to Learning to Rank (LTR)

After retrieval produces a candidate set, **Learning to Rank (LTR)** models re-order results.

The source frames LTR as learning a function \(f(\mathbf{q}, \mathcal{D})\) that produces a ranking (often via scoring each document and sorting by score). Embeddings contribute to LTR by:

- improving candidate set quality (better retrieval recall/semantic matching),
- providing learned semantic features for the ranker,
- enabling multimodal ranking (e.g., combining text and image embeddings).

Cross-reference: [[learning-to-rank]].

---

## Neural ranker architecture spectrum (new)

The new source places two-tower models within a broader set of neural matching/ranking paradigms (query→document, user→item):

- **Two-tower / dual encoder (bi-encoder):**
  - Representation-based; embeddings computed independently, similarity at output (late interaction).
  - Strong serving efficiency; supports decoupled indexing of document/item embeddings.
- **Interaction-focused “matching” models (e.g., DRMM, KNRM):**
  - Build an interaction matrix over words/phrases and apply CNN/MLP-style modeling.
  - More interaction than two-tower, typically more expensive.
- **Cross-encoders (e.g., [[bert]] used as a cross-encoder):**
  - Jointly encode query and document to model full token-level interactions.
  - Usually highest accuracy but expensive; typically used in re-ranking.
- **Late-interaction hybrids (e.g., ColBERT):**
  - Preserve query-document decoupling while allowing richer matching than a single dot product (token-level late interaction).
  - Can still support indexing document-side representations for efficient retrieval.

Cross-references: [[bert]], [[learning-to-rank]], [[dense-retrieval]].

---

## Enhancing two-tower embeddings: interaction and regularization extensions (new)

A recurring limitation of vanilla two-tower models is **limited information exchange between towers** (because embeddings are trained largely independently, interacting only at the final similarity computation).

The new source summarizes several extensions aimed at improving quality while maintaining efficiency:

### Dual Augmented Two-Tower (DAT)
- Augments each tower’s input with additional vectors capturing **historical positive interaction information** from the other tower.
- Adds a **category alignment loss** to address category imbalance by transferring knowledge from high-data categories to low-data categories.
- The source notes later research found gains can be **limited**.

### Interaction Enhanced Two Tower (IntTower)
Designed to improve both **information interaction** and **inference efficiency**, with three main components:

- **Light-SE block (feature recalibration):**
  - Inspired by Squeeze-and-Excitation Networks (SENet).
  - Learns feature importance weights to emphasize informative features and suppress less useful ones.
  - Uses a lightweight single fully-connected layer variant for efficiency.
- **FE-block (Fine-grained & Early feature interaction):**
  - Inspired by ColBERT’s “late interaction” ideas.
  - Performs fine-grained early interactions between multi-layer user representations and the last layer of item representation.
- **CIR module (Contrastive Interaction Regularization):**
  - Uses an InfoNCE-style contrastive loss to bring user vectors closer to positive items (and farther from negatives).
  - Combined with standard supervised loss (e.g., logloss on labels).

Reported results (as described in the source):
- IntTower outperforms baselines including Logistic Regression, Two-Tower, DAT, and COLD for pre-ranking.
- It can be comparable to some heavier ranking models (Wide&Deep, DeepFM, DCN, AutoInt) with negligible parameter/training overhead and acceptable latency.

Cross-references: [[contrastive-learning]], [[learning-to-rank]], [[recommender-systems]].

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

Additional model families/architectures emphasized by the new source (in retrieval + pre-ranking contexts):

- **DSSM-style** two-tower models (dual encoder / bi-encoder) originally developed for mapping queries to relevant documents using clickthrough data.
- **ColBERT-style** late interaction models (token-level late interaction while still enabling document-side indexing).

Cross-references: [[bert]], [[contrastive-learning]], [[multimodal-ml]], [[dense-retrieval]].

---

## Practical notes and caveats

- **Embeddings are infrastructure-dependent in search:**
  - They require a **vector index** and an ANN retrieval layer.
- **Bias and feedback loops:**
  - Changes in embedding retrieval can change what users see, which changes clicks, which can reinforce bias unless corrected with methods like [[unbiased-learning-to-rank]].
- **Page design affects interaction biases:**
  - The source notes eye-tracking evidence that UI design changes can “flatten” attention distribution across ranks, affecting position bias estimators.

### Serving tradeoffs in two-tower systems (new)
- **Decoupling is the core efficiency win:** precompute and index document/item embeddings; compute query/user embedding online.
- **Freezing embeddings trades freshness vs latency/cost:**
  - Freezing item embeddings speeds serving but requires a strategy for updates (incremental embedding refresh, periodic full rebuilds, or offline daily retraining + reindex).
- **Richer interactions usually move later in the cascade:**
  - Cross-encoders and other interaction-heavy architectures tend to be used in re-ranking due to cost.

Cross-references: [[vector-index]], [[approximate-nearest-neighbors]], [[learning-to-rank]], [[position-bias]], [[trust-bias]], [[selection-bias]].

---

## Related pages

- [[learning-to-rank]] (LTR objectives and ranking models)
- [[approximate-nearest-neighbors]] (ANN algorithms for vector search)
- [[contrastive-learning]] (common training paradigm for modern embedding models)
- [[bm25]] and [[tf-idf]] (lexical retrieval baselines used alongside embeddings)
- [[bert]] (text embedding model family)
- [[vector-database]] / [[vector-index]] (storage and serving layer for embeddings)
- [[counterfactual-learning]] and [[unbiased-learning-to-rank]] (learning from biased click signals)
- [[dense-retrieval]] (retrieval with dual encoders / embeddings)
- [[hybrid-retrieval]] (combining lexical and dense methods)
- [[recommender-systems]] (user-item retrieval and ranking pipelines)

---

## Source integration notes

- This page is initialized from a source focused on **Learning to Rank in Web Search**, integrating the parts relevant to embeddings:
  - vector