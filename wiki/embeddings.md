---
slug: embeddings
sources:
- hav4ik.github.io
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
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

### New: Why dense retrieval needs explicit “relevance filtering”
A CIKM’24 paper on product search (Rossi et al., 2024) highlights a practical issue in **embedding-based retrieval**:

- Dense retrieval via [[approximate-nearest-neighbors]] often optimizes **recall**, but can return **low precision** candidate sets (many irrelevant items).
- Unlike lexical retrieval (inverted index keyword matching), ANN-based dense retrieval has **no natural cutoff**: it will always return top-*K* neighbors.
- Raw **cosine similarity** scores are often trained with **contrastive** or **ranking** losses and are therefore:
  - hard to interpret as absolute relevance, and
  - **not comparable across queries** (i.e., a cosine score threshold that works for one query may fail for another).

Implication for system design:
- If retrieval returns too many irrelevant candidates, the reranker must spend compute demoting them—wasting budget and potentially harming latency.
- A lightweight post-retrieval **relevance filter** can reduce the load on re-ranking and improve user experience (especially in product search where “relevant items” may be few).

Cross-references: [[dense-retrieval]], [[hybrid-retrieval]], [[learning-to-rank]].

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

The new CIKM’24 relevance-filtering source adds a second “freezing” pattern:
- For post-retrieval filtering, they **freeze the dual encoders** and train only a lightweight calibration/filtering module on top (the “Cosine Adapter”), potentially using **different training data** than the dual encoder (e.g., human relevance judgments vs engagement logs).

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

### New: Precision control, truncation, and “no natural cutoff” in dense retrieval
Dense retrieval often returns a fixed top-*K* from ANN search. The new source emphasizes:

- Dense retrieval has **no inherent mechanism** (unlike lexical matching) to stop returning candidates when they become irrelevant.
- Using:
  - **top-*K*** alone, or
  - a **global cutoff on raw cosine similarity**
  
  is often insufficient because cosine scores are **not calibrated** and **not comparable across queries**.

This connects to classical IR work on **ranked list truncation** (deciding where to cut off a ranked list), but with different signals:
- Ranked list truncation approaches often model score sequences and document statistics; some use **self-attention over the candidate list**.
- The new approach instead uses the **query embedding** to adapt how to interpret cosine scores for that query.

Cross-references: [[information-retrieval]], [[learning-to-rank]].

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

## New: Relevance filtering for embedding-based retrieval (Cosine Adapter)

A production-motivated approach from Rossi et al. (CIKM’24) introduces a lightweight **relevance filtering** module that sits between ANN retrieval and re-ranking.

### Motivation: interpretability and cross-query score comparability
Key observations from the source:

- Dual-encoder cosine scores are shaped by **relative** objectives:
  - contrastive loss with in-batch negatives, or
  - listwise softmax ranking losses.
- These objectives encourage ordering *within a query* (positive > negatives), but do not make cosine similarity an **absolute relevance score**.
- Therefore:
  - “cosine similarity scores cannot be compared across different queries,” and
  - global thresholds on raw cosine similarity often behave poorly (for many queries: filter everything or nothing).

**This strengthens and partially contradicts the simplistic operational idea** that “a cosine threshold” is a universally meaningful relevance cutoff:
- Existing page framing: cosine similarity is used for nearest neighbor retrieval and can be thresholded.
- New source: cosine similarity is often **not interpretable/calibrated** for cross-query thresholding.
- Resolution: cosine similarity works well for **ranking neighbors**, but may need **calibration** for **global filtering/truncation** decisions.

Cross-references: [[dense-retrieval]], [[contrastive-learning]], [[learning-to-rank]].

### Approach: query-dependent monotonic mapping of cosine scores
They propose learning a function that transforms raw cosine similarity \(x=\cos(q,p)\in[-1,1]\) into an interpretable relevance score:

\[
\tilde{P}_i = \{p_j \mid F_{\Theta}( \cos(q_i,p_j)) \ge t \}
\]

- \(F_\Theta(\cdot)\): **monotonic** mapping to preserve the relative order of candidates (minimizing disruption to recall).
- \(\Theta\): **query-dependent parameters** predicted from the **query embedding**.
- \(t\): a **global threshold** tuned offline.

This component is called the **Cosine Adapter**:
- Input: query embedding (from the frozen dual encoder).
- Output: parameters \(\Theta\) for the chosen mapping function \(F\).
- Training: binary classification objective (binary cross entropy) to predict relevance probability after a sigmoid:
  - \(F\) outputs a logit; \(\sigma(F)\) is treated as \(P(\text{relevant}\mid q,p)\).

Cross-references: [[representation-learning]], [[information-retrieval]].

### Mapping function family (calibration functions)
They explore several monotonic transformation families:

- Raw: \(F(x)=x\)
- Linear: \(F(x\mid a,b)=ax+b\)
- Square root: \(F(x\mid a,b)=\operatorname{sgn}(x)\,a\sqrt{|x|}+b\)
- Quadratic: \(F(x\mid a,b)=\operatorname{sgn}(x)\,a x^2+b\)
- Power: \(F(x\mid a,b,k)=\operatorname{sgn}(x)\,a|x|^k+b,\quad k\in(0,2)\)

Important serving property:
- Adapter compute is **once per query** (to produce \(\Theta\)), plus **\(O(K)\)** per candidate list to compute calibrated scores.
- The paper contrasts this to list-truncation methods that do self-attention over candidates with **\(O(K^2 d)\)** complexity.

Cross-references: [[neural-networks]] (if present), [[search]].

### Training and deployment pattern: “freeze encoders, train the adapter”
The method decouples concerns:

- **Dual encoders** are trained (contrastive or listwise) for good retrieval.
- **Cosine Adapter** is trained on relevance-labeled data to calibrate/filter results.
- The dual encoder is **frozen** while training the adapter.

**Practical note:** the adapter training data can be different:
- Dual encoders may train on engagement/click signals.
- Adapter may train on **human relevance judgments** (as in their Walmart experiments).

Cross-references: [[counterfactual-learning]] (for bias/noisy click labels), [[learning-to-rank]].

### Metrics and tradeoffs (precision vs recall)
Filtering introduces an explicit tradeoff:
- Goal: significantly improve **precision** with only small loss of **recall**.

Metrics used in the paper include:
- **PR AUC** (no filtering applied; measures separability of relevant vs irrelevant)
- **P@R95** (precision at 95% recall relative to no filtering; uses a global threshold)
- **Filter%** (how many results removed)
- **Null%** (queries returning zero results after filtering)
- **MRR** (reported on MS MARCO)

This makes “retrieval set quality” measurable beyond recall-only thinking.

Cross-references: [[evaluation-metrics]] (if present), [[information-retrieval]].

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
\mathbb{1}_{y_1 \ne y_2} \, \