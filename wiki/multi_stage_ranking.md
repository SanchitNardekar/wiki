---
slug: multi_stage_ranking
sources:
- hav4ik.github.io
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
tags: []
title: Multi-Stage Ranking Pipelines
updated: '2026-04-14'
---

# Multi-Stage Ranking Pipelines

Multi-stage ranking pipelines are the standard production architecture for web search and large-scale recommendation/search systems. Instead of running an expensive, highly accurate ranker over *all* documents, the system progressively narrows down a huge candidate set through multiple stages—each stage trading off **cost vs. quality**—until a final ranked list is produced.

This page focuses on the ranking pipeline shape (retrieval → filtering → re-ranking), and how [[learning_to_rank]] methods and evaluation metrics like [[ndcg]] fit into these stages.

---

## Why multi-stage pipelines exist

At web scale, it is computationally infeasible to compute rich features and run heavy ML models over the entire corpus for every query. A multi-stage pipeline:

- **Reduces compute** by aggressively shrinking the candidate set early.
- **Improves latency predictability** by reserving expensive models for only top candidates.
- **Improves quality** by allowing later stages to use richer features and more powerful rankers.

**Additional motivation from industry reports:** strict latency constraints strongly shape stage design; even ~100ms latency regressions have been reported to measurably degrade user experience and revenue in large online systems. This reinforces why “one big model” is rarely used end-to-end in production.

Cross-reference: [[latency]], [[online_experimentation]].

---

## High-level pipeline (retrieval → ranking)

A common abstract schematic is:

1. **Offline indexing** (continuous, not on the critical query path)
2. **Top-*k* retrieval** (fast candidate generation; sometimes called “Level-0 ranking” or “matching”)
3. **Ranking / re-ranking** (ML-based scoring and sorting of retrieved candidates)

The core online flow is:

- Given a query **q**, retrieve a set of candidate documents **D**.
- Compute features for each (q, d) pair.
- Produce scores **sᵢ = f(q, dᵢ)** and sort by score.

### Cascade / funnel framing (recall → pre-ranking → ranking → re-ranking)

Many large systems describe the same architecture as a **cascade ranking system**:

- **Recall / Retrieval**: maximize recall with very fast matching
- **Pre-ranking**: fast ML scoring over a *larger* candidate set than later stages
- **Ranking**: heavier models (often deep neural nets) over fewer candidates
- **Re-ranking**: post-processing, diversification, business rules, or even heavier models

This naming emphasizes that early stages are often optimized for **recall-like objectives** while later stages focus on **final ordering quality** and richer constraints.

Cross-reference: [[candidate_generation]], [[retrieval]], [[re_ranking]], [[diversification]].

---

## Stage 0: Offline indexing (continuous)

Indexing is typically performed **offline** and continuously. Modern systems may maintain multiple indexes:

- **Inverted index** (posting lists)
  - Maps terms → documents
  - Supports term-based scoring like **TF-IDF** and **BM25**
- **Vector index**
  - Stores **embeddings** for documents (and/or queries)
  - Used for nearest-neighbor retrieval by cosine/euclidean similarity
  - Embeddings often come from contrastive-learning models (e.g., “BERT-like” for text; vision-text models for visual search)
- **Feature index**
  - Stores many engineered signals and compressed learned features
  - These are primarily used in later re-ranking stages

**Key point:** Feature engineering remains central—“the more expressive your features are, the better your ranking layer will perform.”

### Embedding indexes from pre-ranking models (two-tower / dual-encoder)

In many recommender systems and ads/search stacks, a key offline artifact is an **embedding index** produced by a **two-tower (dual-encoder / bi-encoder)** model:

- A **query/user tower** encodes the request/context into an embedding.
- A **document/item/ad tower** encodes candidates into embeddings.
- Similarity is computed via **late interaction** (often a dot product).

A common production pattern is:

- Precompute and store **item tower embeddings** in an indexing service (vector DB / ANN structure).
- At inference, compute query/user embedding online and retrieve nearest neighbors quickly.
- In some deployments, embeddings (especially the item tower) can be **frozen after training** and updated on a schedule; some systems even freeze both towers and rebuild indexes periodically (e.g., daily offline retrains in certain large-scale deployments).

Cross-reference: [[two_tower_model]], [[dual_encoder]], [[dense_retrieval]], [[vector_search]], [[embeddings]].

Cross-reference: [[indexing]], [[inverted_index]], [[vector_search]], [[bm25]], [[embeddings]].

---

## Stage 1: Top-*k* retrieval (candidate generation / “matching”)

**Top-*k* retrieval** selects potentially relevant documents for a query.

Typical approaches:

- **Keyword / entity matching** via the inverted index
- **Embedding-based retrieval** via vector search (kNN over document embeddings)
- **Hybrid retrieval** combining both sparse (lexical) and dense (embedding) signals
- Optional **query expansion** (e.g., knowledge-graph-based) to retrieve more relevant candidates

### Approximate nearest neighbors (ANN) for scale

Large-scale systems usually avoid exact metric trees (e.g., k-d trees) due to speed/memory tradeoffs at scale. Instead they use **Approximate Nearest Neighbors (ANN)** methods (e.g., hashing-based approaches) to achieve near-constant-time retrieval behavior.

### Dense retrieval has “no natural cutoff” (precision risk)

A key practical difference between lexical retrieval and ANN-based dense retrieval (from Walmart’s CIKM’24 report on relevance filtering):

- **Lexical retrieval** inherently limits the retrieved set via term constraints (keyword matching).
- **Dense ANN retrieval** can always return many “nearest” items, even when **very few (or zero) truly relevant items exist** for the query.
  - In product search, the number of relevant products is often small; surfacing a long tail of irrelevant items can harm user experience even if top ranks contain relevant items.

This motivates adding an explicit **relevance filtering** step between ANN retrieval and downstream re-ranking.

Cross-reference: [[approximate_nearest_neighbor]], [[dense_retrieval]], [[vector_search]], [[hybrid_retrieval]].

### Where two-tower models fit: retrieval vs pre-ranking

The existing page emphasized that **two-tower models are a go-to architecture for “pre-ranking”**. The new source adds an additional nuance:

- In embedding-based retrieval, the **dual encoder** produces embeddings and ANN returns **top-K** based on cosine similarity.
- However, **top-K alone is often insufficient** to remove irrelevant candidates because:
  - There’s no “natural” cutoff in dense retrieval.
  - Cosine similarity is often trained with contrastive / ranking losses, making it **hard to interpret as an absolute relevance score** and **not comparable across queries**.

Operationally, this supports the idea that “retrieval” often needs an additional lightweight stage for **candidate set shaping** (filtering/truncation/calibration) before expensive ranking.

Cross-reference: [[two_tower_model]], [[dual_encoder]].

---

## Stage 1.5 (new): Relevance filtering / candidate set truncation for dense retrieval

Many production pipelines include an intermediate stage—after retrieval but before full ranking—to **reduce the burden on the reranker** and improve precision. The new source describes this explicitly for embedding-based retrieval:

- Goal: Filter out obviously irrelevant ANN results **before** sending candidates to the reranker.
- Rationale: If a query has only 1 relevant item, retrieving K=1000 will include ~999 irrelevant items; making the reranker fix this is wasteful and can degrade latency/cost.

This stage is closely related to:
- “Pre-ranking” (fast scoring over many candidates)
- “Relevance control” / “precision gating”
- “Ranked list truncation” (deciding where to cut off a retrieved list)

Cross-reference: [[re_ranking]], [[candidate_generation]].

### Why naive cosine thresholds or top-K truncation can fail

The Walmart report argues:

- A global threshold on **raw cosine similarity** is not optimal because cosine scores are:
  - often **not calibrated** to be probabilistic,
  - trained for **relative** ordering within a query, and
  - **not comparable across queries**, so a single global cutoff behaves inconsistently.

This is especially pronounced when dense retrieval models are trained with:
- **contrastive loss** with in-batch negatives, or
- **softmax listwise losses**,

because both shape embedding spaces based on **relative distances**, not absolute relevance.

Cross-reference: [[contrastive_learning]], [[infonce]].

### Cosine Adapter (query-dependent calibration of cosine similarity)

The paper introduces a lightweight module called **Cosine Adapter**:

- Input: the **query embedding** produced by the frozen dual encoder.
- Output: parameters **Θ** that define a **query-dependent** monotonic mapping function **FΘ(x)** applied to the raw cosine similarity $x = \cos(q, p)$.
- Filtering rule:

$$
\tilde{P_i} = \{p_j \mid F_{\Theta}(\cos(q_i, p_j)) \ge t\}
$$

where $t$ is a **global threshold** learned/tuned offline.

**Key design choices:**
- Mapping functions are chosen to be **monotonic** to preserve ordering as much as possible (minimizing recall loss), while enabling calibration.
- Several function families were explored:
  - raw: $F(x)=x$
  - linear: $F(x)=ax+b$
  - square root: $F(x)=\mathrm{sgn}(x)\,a\sqrt{|x|}+b$
  - quadratic: $F(x)=\mathrm{sgn}(x)\,ax^2+b$
  - power: $F(x)=\mathrm{sgn}(x)\,a|x|^k+b$, with $k\in(0,2)$

**Training objective:**
- Train adapter with **binary cross-entropy** on relevance labels:
  - interpret $\sigma(F)$ as the probability the pair is relevant.
- **Dual encoder is frozen** during adapter training.
- Adapter may be trained on a **different dataset** than the dual encoder (e.g., dual encoder trained on engagement logs; adapter trained on human judgments).

Cross-reference: [[calibration]] (if it exists), [[dual_encoder]].

### Computational profile and why this is “pipeline-friendly”

At inference time:

- Adapter feed-forward cost: $O(d^2)$ per query (run once), where $d$ is embedding dimension.
- Score mapping cost: $O(K)$ (a few ops per candidate).
- Compared to ranked list truncation approaches using self-attention over candidates (e.g., Choppy-style), which can be $O(K^2 d)$, this is much cheaper.

This fits multi-stage pipelines as an “in-between” stage that improves end-to-end system efficiency by reducing candidates early.

Cross-reference: [[latency]].

### Metrics for filtering stages (precision/recall trade-offs)

The new source emphasizes that introducing filtering changes retrieval behavior from “maximize recall” toward an explicit **precision–recall trade-off** and proposes tracking additional metrics:

- **PR AUC** (area under precision–recall curve) — evaluated without committing to a single threshold.
- **P@R95**: precision at a threshold chosen to achieve **95% recall** relative to no filtering.
- **Filter%**: percentage of retrieved results removed.
- **Null%**: percentage of queries that return **zero** results after filtering.
- **MRR**: still reported for MS MARCO because it is a standard retrieval metric.

These metrics complement traditional offline ranking metrics such as [[mrr]], [[map]], and [[ndcg]] and are especially relevant for retrieval and candidate-set shaping stages.

Cross-reference: [[offline_evaluation]] (if it exists), [[mrr]].

### Empirical outcomes (MS MARCO + Walmart product search + online A/B)

Key reported findings:

- On **MS MARCO passage retrieval** (evaluated at K=10 and K=1000):
  - Calibrated mappings improve PR AUC and improve **P@R95** versus raw cosine thresholding.
  - For K=1000, calibrated methods reduce **Null%** compared to raw-score thresholding (raw threshold tends to keep all or discard all for many queries).
  - Ranked-list truncation baseline (Choppy) can truncate too aggressively, yielding low recall in this dataset (where many queries have only one relevant passage).

- On **Walmart product search** (human-judged labels: exact/substitute/irrelevant):
  - Cosine Adapter improves PR AUC and P@R95 over raw cosine thresholds.
  - Adapter behavior interacts with how the dual encoder was trained (contrastive vs listwise). (See contradictions/notes below.)

- **Online A/B test (Walmart.com)**:
  - Integrated into embedding retrieval → relevance filter → downstream reranker.
  - Threshold tuned for **99% recall** to minimize risk.
  - Reported improvement in precision for impacted queries (e.g., top-5 and top-10 precision lift), while orders and GMV were neutral (no statistically significant negative impact).

Cross-reference: [[online_experimentation]].

### Failure modes / query patterns that lose recall

Because filtering uses a global threshold tuned for high recall (e.g., 95% or 99%), some relevant items are filtered out. The paper reports filtered-out relevant items correlate with:

- rare words / rare brands
- misspellings
- numbers

This resembles common retrieval brittleness patterns in dense retrieval and suggests monitoring for tail-query regressions when adding filtering.

Cross-reference: [[query_understanding]] (if it exists), [[dense_retrieval]].

---

## Stage 2+: Ranking and re-ranking (Learning to Rank)

After retrieval, candidates are **scored and sorted** by relevance (and optionally user preferences). For small systems, hand-crafted rules can be sufficient; for large systems, **Machine Learning / Learning to Rank (LTR)** is typically used.

### Formal problem statement

Given:

- query **q**
- retrieved documents **D = {d₁, …, dₙ}**

learn a function:

- **f(q, D)** that induces an ordering
- commonly implemented as scoring each document: **sᵢ = f(q, dᵢ)**, then sorting by **sᵢ**

Cross-reference: [[learning_to_rank]].

### Neural architecture spectrum across stages (representation vs interaction)

The new source highlights a commonly used taxonomy of neural ranker/matching architectures that maps cleanly to multi-stage pipelines:

- **Representation-based (decoupled) models**: e.g., **Two-Tower / dual-encoder**
  - Compute query and doc embeddings independently
  - Interaction is “late” (e.g., dot product)
  - Enables embedding indexes and very fast serving
  - Common in retrieval and pre-ranking
- **Interaction-based models**: e.g., DRMM/KNRM-style interaction matrices
  - Model fine-grained term/phrase interactions earlier in the network
  - Typically heavier than pure dual encoders
- **Cross-encoders**: e.g., BERT cross-encoder scoring (jointly encoding query+doc)
  - Most expressive for query-document interactions
  - Usually reserved for later ranking/re-ranking due to cost
- **Late-interaction hybrids**: e.g., ColBERT-style approaches
  - Preserve decoupling but allow richer late interaction than a single dot product

**Operational takeaway:** multi-stage pipelines often progress from **decoupled/cheap** (representation learning + ANN) → **more interaction/cost** (cross-encoders or interaction-heavy models).

Cross-reference: [[bert_ranking]], [[cross_encoder]], [[colbert]], [[neural_ir]].

---

## LTR training paradigms used in ranking stages

LTR methods are often described along two axes:

### Offline vs online

- **Offline LTR**
  - Train once (or periodically) on a fixed dataset.
- **Online LTR**
  - Learn from user interactions in real time; update after interactions.

### Supervised vs counterfactual

- **Supervised LTR**
  - Uses human-judged relevance labels (often graded, e.g., 1–5).
  - Optimization objectives can be:
    - **Pointwise** (single document)
    - **Pairwise** (document pairs)
    - **Listwise** (entire list)
- **Counterfactual LTR / Unbiased LTR**
  - Uses historical interactions (clicks, conversions, engagement)
  - Must correct for biases in implicit feedback

Cross-reference: [[unbiased_learning_to_rank]], [[counterfactual_learning]], [[online_learning_to_rank]].

---

## Ranking quality metrics (offline evaluation)

Common information retrieval ranking metrics include:

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **ERR** (Expected Reciprocal Rank)
- **NDCG@T** (Normalized Discounted Cumulative Gain at cutoff T)

The source notes that MAP and MRR are widely used for retrieval but are less preferred for *graded relevance* ranking because they don’t directly incorporate relevance levels (unless modified or labels are binary). NDCG is highlighted as the most commonly used metric for graded relevance ranking.

### NDCG definition (graded relevance)

For cutoff **T**, DCG is:

$$
DCG@T = \sum_{i=1}^{T} \frac{2^{l_i} - 1}{\log(1+i)}
$$

and

$$
NDCG@T = \frac{DCG@T}{\max DCG@T}
$$

where $l_i$ is the relevance label of the document at rank $i$.

Cross-reference: [[ndcg]], [[dcg]], [[err]], [[mrr]], [[map]].

---

## Common supervised rankers used in re-ranking stages

Multi-stage pipelines often use a **cheap model early** (e.g., linear/GBDT) and **heavier models later** (e.g., neural models), but the source emphasizes several classical LTR methods that remain strong in practice.

### RankNet (pairwise)

- Models probability that document i should rank above j:

$$
P_{ij}=\frac{1}{1+e^{-\sigma(s_i-s_j)}}
$$

- Optimizes cross-entropy on pairwise preferences.

Cross-reference: [[ranknet]].

### ListNet (listwise)

- Defines a listwise probability distribution over rankings (Plackett–Luce style)
- Optimizes cross-entropy between distributions induced by labels vs model scores

Cross-reference: [[listnet]].

### LambdaRank and LambdaMART (metric-aware gradients)

Problem: metrics like NDCG/ERR are not directly differentiable due to sorting.

- **LambdaRank** modifies gradients by weighting pairwise updates with the magnitude of metric change (e.g., $|\Delta NDCG_{ij}|$) if two documents swap positions.
- **LambdaMART** combines LambdaRank’s idea with gradient boosted decision trees (“Multiple Additive Regression Trees”).