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

## Deep metric learning (DML) as a lens on embeddings (new)

A complementary view of embeddings—especially common in **visual search** and some retrieval settings—is **Deep Metric Learning**: learning an embedding function \(f_\theta(\cdot)\) such that *distances in embedding space* reflect label-based similarity.

### Supervised metric learning problem setting
Given data points \(\mathcal{X}\), labels \(\mathcal{Y}\) (finite discrete set), an embedding network:

- \(f_\theta: \mathcal{X} \to \mathbb{R}^n\)

and a (usually fixed) distance \(\mathcal{D}: \mathbb{R}^n \to \mathbb{R}\), train so that:

- \(\mathcal{D}(f_\theta(x_1), f_\theta(x_2))\) is **small** when \(y_1=y_2\)
- and **large** when \(y_1\neq y_2\)

This perspective connects directly to retrieval: nearest neighbors under \(\mathcal{D}\) define candidate matches.

Cross-references: [[representation-learning]], [[information-retrieval]], [[nearest-neighbor-search]].

### Distances used: Euclidean vs cosine (and normalization)
The DML survey emphasizes:
- “Direct” contrastive approaches commonly use **\(L_2\) (Euclidean) distance**.
- More recent “angular margin” approaches operate in **cosine/angle space** by:
  - normalizing feature vectors (embeddings) to unit length and
  - normalizing classifier weights (class prototypes), then
  - adding explicit **margins** in cosine or angle space.

**Connection to IR embeddings (note):**
- The existing page already frames dense retrieval with **cosine similarity** or **Euclidean distance**. The DML survey adds a practical training nuance: cosine/angle objectives typically rely on explicit **normalization** (unit vectors) and margin shaping, which affects downstream distance geometry.

Cross-references: [[contrastive-learning]].

---

## Loss functions for learning embeddings (expanded with DML survey)

Embeddings are shaped primarily by the **training objective**. In retrieval systems, these objectives are often formulated as pairwise/listwise ranking losses (see [[learning-to-rank]]) or contrastive objectives (see [[contrastive-learning]]). In supervised metric learning, they are often expressed directly in terms of distances/margins.

### Contrastive loss (pairwise)
Classic contrastive loss (Chopra et al., 2005) for two samples \((x_1,x_2)\):

\[
\mathcal{L}_\text{contrast}
=
\mathbb{1}_{y_1 = y_2} \, \mathcal{D}^2(f_\theta(x_1), f_\theta(x_2))
+
\mathbb{1}_{y_1 \ne y_2} \, \max\left(0, \alpha - \mathcal{D}^2(f_\theta(x_1), f_\theta(x_2))\right)
\]

- \(\alpha\) is a **margin** to prevent collapse (mapping everything to the same point).

Cross-references: [[contrastive-learning]].

### Triplet loss and negative mining
Triplet loss (Schroff et al., 2015) uses an anchor/positive/negative \((x_a,x_p,x_n)\):

\[
\mathcal{L}_\text{triplet}
=
\max\left(0, \mathcal{D}^2(f_\theta(x_a), f_\theta(x_p)) - \mathcal{D}^2(f_\theta(x_a), f_\theta(x_n)) + \alpha \right)
\]

Key practical ingredient emphasized by the DML survey:
- **Negative sample mining** is often required to pick informative/hard negatives.

Cross-references: [[contrastive-learning]].

### Limitations highlighted for distance-based contrastive objectives (new)
The DML survey argues many “direct” Euclidean contrastive objectives face two recurring issues:

- **Expansion issue:** hard to ensure *all* same-class samples collapse to a coherent region globally (objectives can be too local/batch-dependent).
- **Sampling issue:** strong reliance on sophisticated mining (hard in distributed training).

**Integration note:** This frames why many modern systems prefer objectives that use *global class prototypes* (e.g., softmax-based angular margins) or batch-contrastive methods (e.g., InfoNCE variants).

Cross-references: [[contrastive-learning]].

### Center loss: augmenting softmax with embedding compactness (new)
Center loss (Wen et al., 2016) adds a term pulling features toward per-class centers \(c_{y_i}\):

\[
\mathcal{L}_\text{center} = \mathcal{L}_\text{softmax} + \frac{\lambda}{2}\sum_{i=1}^{N}\|z_i - c_{y_i}\|_2^2
\]

- Addresses:
  - sampling burden (less mining),
  - tighter intra-class clusters.

Cross-references: [[representation-learning]].

### Angular margin softmax family (CosFace / ArcFace etc.) (new)
The DML survey describes a shift toward **angular margin** methods (popular in face recognition and instance retrieval):

Common setup:
- Normalize embeddings \(z\) and class weights \(W_j\) so \(\|z\|=\|W_j\|=1\).
- Logits become cosine similarities: \(W_j^\top z = \cos\theta_j\).
- Introduce a **margin** to enforce stronger inter-class separation.

#### SphereFace (multiplicative angular margin)
SphereFace (Liu et al., 2017) uses a multiplicative angular margin \(\mu\) inside \(\cos(\mu \theta)\).
- The survey notes optimization difficulties due to cosine non-monotonicity and the need for piecewise tricks.

#### CosFace (additive cosine margin)
CosFace (Wang et al., 2018):

\[
\mathcal{L}_\text{CosFace}
=
-\frac{1}{N}\sum_i
\log
\frac{\exp\{s(\cos(\theta_{y_i,i})-m)\}}
{\exp\{s(\cos(\theta_{y_i,i})-m)\}+\sum_{j\ne y_i}\exp\{s\cos(\theta_{j,i})\}}
\]

- \(s\): scaling parameter; \(m\): cosine margin parameter.

#### ArcFace (additive angular margin)
ArcFace (Deng et al., 2019) adds margin in **angle space**:

\[
\mathcal{L}_\text{ArcFace}
=
-\frac{1}{N}\sum_i
\log
\frac{\exp\{s\cos(\theta_{y_i,i}+m)\}}
{\exp\{s\cos(\theta_{y_i,i}+m)\}+\sum_{j\ne y_i}\exp\{s\cos(\theta_{j,i})\}}
\]

- Reported as slightly stronger than CosFace across benchmarks in the survey.

#### Hyperparameter sensitivity: scaling \(s\) and margin \(m\)
The DML survey emphasizes:
- Choosing \(s\) and \(m\) is crucial; poor choices can make training under-penalize errors or over-penalize confident correct cases.
- AdaCos proposes a fixed scaling heuristic: \(\tilde{s} \approx \sqrt{2}\log(C-1)\) where \(C\) is number of classes.

**Practical note (new, and a mild contradiction to “contrastive learning usually computes embeddings”):**
- The existing page says embeddings are “usually computed using state-of-the-art contrastive learning neural networks.”  
- The DML survey suggests in *supervised metric learning*, **softmax + margin** objectives (ArcFace/CosFace variants) can outperform direct contrastive losses (triplet/contrastive), and are widely used in strong real-world retrieval/recognition solutions.  
  - **Resolution:** both families are common; “usually” depends heavily on modality, supervision type, and whether training is framed as classification vs pairwise ranking/contrastive.

Cross-references: [[contrastive-learning]], [[multimodal-ml]].

### Sub-centers and dynamic margins for long-tail / noisy classes (new)
The DML survey highlights extensions for difficult, imbalanced, and noisy datasets:

- **Sub-Center ArcFace:** multiple centers per class; uses nearest sub-center, helping with intra-class variability and label noise.
- **Dynamic margin:** class-dependent margins based on class frequency (rarer classes get larger margins), improving convergence under imbalance.

**Connection to IR:** This is relevant to retrieval domains with:
- long-tail entities/items,
- noisy labels (click logs, weak supervision),
- heterogeneous class granularity (e.g., product catalogs).

Cross-references: [[recommender-systems]], [[counterfactual-learning]] (for noisy/biased labels).

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

The source frames LTR as learning a function \(f(\mathbf{q}, \mathcal{D})\) that produces a ranking (often via scoring each document and sorting by score). Embeddings contribute to LTR