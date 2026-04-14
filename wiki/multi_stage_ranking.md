---
slug: multi_stage_ranking
sources:
- hav4ik.github.io
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

Cross-reference: [[approximate_nearest_neighbor]], [[candidate_generation]], [[dense_retrieval]], [[hybrid_retrieval]].

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

\[
DCG@T = \sum_{i=1}^{T} \frac{2^{l_i} - 1}{\log(1+i)}
\]

and

\[
NDCG@T = \frac{DCG@T}{\max DCG@T}
\]

where \(l_i\) is the relevance label of the document at rank \(i\).

Cross-reference: [[ndcg]], [[dcg]], [[err]], [[mrr]], [[map]].

---

## Common supervised rankers used in re-ranking stages

Multi-stage pipelines often use a **cheap model early** (e.g., linear/GBDT) and **heavier models later** (e.g., neural models), but the source emphasizes several classical LTR methods that remain strong in practice.

### RankNet (pairwise)

- Models probability that document i should rank above j:

\[
P_{ij}=\frac{1}{1+e^{-\sigma(s_i-s_j)}}
\]

- Optimizes cross-entropy on pairwise preferences.

Cross-reference: [[ranknet]].

### ListNet (listwise)

- Defines a listwise probability distribution over rankings (Plackett–Luce style)
- Optimizes cross-entropy between distributions induced by labels vs model scores

Cross-reference: [[listnet]].

### LambdaRank and LambdaMART (metric-aware gradients)

Problem: metrics like NDCG/ERR are not directly differentiable due to sorting.

- **LambdaRank** modifies gradients by weighting pairwise updates with the magnitude of metric change (e.g., \(|\Delta NDCG_{ij}|\)) if two documents swap positions.
- **LambdaMART** combines LambdaRank’s idea with gradient boosted decision trees (“Multiple Additive Regression Trees”).

Notes from the source:

- LambdaMART remains a very strong baseline and can outperform newer methods on some benchmarks.
- Computing \(\Delta NDCG\) over all pairs is \(O(n^2)\) per query (naively); similar pairwise metric computations are quadratic.

Cross-reference: [[lambdarank]], [[lambdamart]], [[lightgbm]].

---

## Example: training a LambdaMART stage with LightGBM

The source provides an illustrative training recipe using:

- **LightGBM** objective: `lambdarank`
- Metric: `ndcg`
- Evaluation cutoffs: e.g. `ndcg@1,3,5,10`
- Group/query sizes are required (query grouping is fundamental in LTR datasets)

Dataset example:

- **MSLR-WEB30K**
  - ~3.7M documents
  - ~30k queries
  - 136 features per (q, d)
  - graded relevance labels 0–4

This fits multi-stage pipelines as a typical *re-ranking model* trained offline and deployed online as a scoring stage.

Cross-reference: [[mslr_web30k]], [[ltr_datasets]].

---

## Feature types commonly used by ranking stages

From the LightGBM feature-importance discussion (MSLR-WEB30K example), prominent feature families include:

- **Graph / authority signals** (e.g., PageRank-related features)
- **URL structure features** (e.g., URL length, number of slashes)
- **Document quality classifier outputs** (quality scores)
- **Lexical matching scores** (e.g., Title BM25, body BM25)
- **Query–URL click signals** (historical click count)

This illustrates a common pipeline pattern:

- Early retrieval: sparse/dense matching signals
- Later re-ranking: rich engineered and learned features, including interaction logs

Cross-reference: [[pagerank]], [[click_signals]], [[bm25]].

---

## Unbiased (counterfactual) learning in later stages

When training rankers on clicks or engagement, implicit feedback is biased. The source highlights common biases:

- **Position bias**: top-ranked items get examined/clicked more
- **Selection bias**: some items may have near-zero chance of being examined (e.g., users rarely go to page 2+)
- **Trust bias**: users trust the system and click higher results even if not relevant

**Operational implication:** UI/design changes can shift biases (e.g., eye-tracking heatmaps show distribution of attention changes with SERP design), so bias estimators may need recalibration after major layout changes.

Counterfactual LTR centers around:

- **Counterfactual evaluation**: evaluating a new ranking policy using interaction data gathered under a deployed policy (“behavior policy”)

Cross-reference: [[position_bias]], [[trust_bias]], [[selection_bias]], [[propensity_scoring]], [[counterfactual_evaluation]].

---

## Contradictions / notes

- **No contradictions with existing page content**: this is a new page, so there is no prior text to conflict with.
- **Potential terminology mismatch (note, not a contradiction):**
  - The source sometimes uses “Top-k Retrieval” as “Level-0 Ranking” or “Matching”. In some organizations, “ranking” is reserved for LTR-based scoring stages only, and retrieval is treated as separate from ranking. This page treats retrieval as the first stage in a multi-stage ranking pipeline, consistent with the “Level-0 ranking” framing.

---

## See also

- [[learning_to_rank]]
- [[unbiased_learning_to_rank]]
- [[counterfactual_learning]]
- [[ndcg]]
- [[bm25]]
- [[approximate_nearest_neighbor]]
- [[lightgbm]]
- [[lambdamart]]
- [[candidate_generation]]