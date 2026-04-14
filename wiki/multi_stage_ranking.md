---
slug: multi_stage_ranking
sources:
- hav4ik.github.io
- blog.reachsumit.com
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

### Where two-tower models fit: retrieval vs pre-ranking

The new source emphasizes that **two-tower models are a go-to architecture for “pre-ranking”** in industry. In practice, two-tower models can appear in two closely related ways:

- **Dense retrieval**: two-tower produces embeddings used directly for ANN retrieval (often viewed as part of Stage 1).
- **Pre-ranking**: two-tower computes a fast similarity score (dot product) over a recalled candidate set (often viewed as a separate stage between retrieval and full ranking).

These are compatible: whether you call it “retrieval” or “pre-ranking” often depends on where you draw the boundary between “candidate generation” and “scoring.”

Cross-reference: [[approximate_nearest_neighbor]], [[candidate_generation]], [[dense_retrieval]], [[hybrid_retrieval]], [[two_tower_model]].

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

## Pre-ranking with Two-Tower models (and extensions)

The new source adds detail on the *pre-ranking* stage and why two-tower models are popular there:

### Why two-tower is common in pre-ranking

- **Inference efficiency by design**: query/user and item/doc embeddings computed independently; interact only at output (dot product).
- **Parallelizable towers**: can compute both sides independently (and often cache or precompute item embeddings).
- **Index-friendly**: item tower embeddings can be stored and retrieved via ANN structures.

Cross-reference: [[two_tower_model]], [[approximate_nearest_neighbor]].

### Dual encoder variants: Siamese vs asymmetric (and a reported quality pitfall)

Dual encoders (two-tower) can be structured as:

- **Siamese Dual Encoder (SDE)**: two identical sub-networks, often sharing parameters.
- **Asymmetric Dual Encoder (ADE)**: two distinct encoders.

The new source reports findings (in question-answering retrieval) that:

- SDE can perform significantly better than ADE because ADE may embed inputs into **disjoint embedding spaces**, hurting retrieval quality.
- ADE can be improved by **sharing a projection layer** (ADE-SPL), potentially matching or exceeding SDE.
- Sharing token embedders (ADE-STE) or freezing token embedders (ADE-FTE) yields only marginal improvements.

Cross-reference: [[dual_encoder]], [[siamese_networks]].

### Limitations: lack of cross-tower interaction

A frequently cited limitation of pure two-tower models is **limited query–document (user–item) interaction**, since most computation happens independently per tower and only a simple similarity is computed at the end. This can cap effectiveness compared to interaction-heavy ranking models.

This limitation motivates:

- Later-stage cross-encoders / interaction models in multi-stage pipelines.
- Two-tower *extensions* that inject limited interaction while preserving efficiency.

### Extensions aimed at better interactions while preserving efficiency

The source describes several research directions:

- **DAT (Dual Augmented Two-Tower)**: augment each tower’s input embedding with vectors summarizing historical positive interactions from the other side; may also incorporate category-alignment losses for imbalance.
  - Note: later work reported gains can be limited.
- **IntTower (Interaction Enhanced Two-Tower)**: adds lightweight blocks to increase feature interaction while keeping latency acceptable:
  - **Light-SE block**: a lightweight feature recalibration mechanism inspired by Squeeze-and-Excitation (SENet), to weight/refine features per tower.
  - **FE-block**: fine-grained and early feature interaction inspired by ColBERT-style late interaction ideas.
  - **CIR module**: contrastive interaction regularization using **InfoNCE** loss, combined with logloss.
  - Reported outcome: outperforms baselines like LR and vanilla two-tower; can be comparable to heavier ranking models (e.g., Wide&Deep/DeepFM/DCN/AutoInt) while adding negligible parameters/training time and acceptable serving latency.
- **Single-tower pre-ranking alternatives** (more interaction, less decoupling):
  - Can improve accuracy but break “user-item decoupling,” so need optimization tricks to mitigate efficiency costs.
  - Example ideas include compute-aware feature selection (optimize QPS/RT tradeoffs) or learnable feature selection via dropout-like regularization.

Cross-reference: [[contrastive_learning]], [[infonce]], [[colbert]], [[wide_and_deep]], [[deepfm]], [[dcn]], [[autoint]], [[feature_selection]].

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

- **Potential contradiction / terminology tension (explicit):**
  - Existing page content frames the candidate generation stage as “Top-*k* retrieval” and discusses embedding-based retrieval there.
  - The new source states that **two-tower models are “the current go-to state-of-the-art solution for pre-ranking tasks”**, placing them *after* recall/retrieval rather than as the retrieval mechanism itself.
  - Resolution: in practice, two-tower models are used both ways:
    - as **dense retrieval** (ANN over item embeddings), and/or
    - as a **pre-ranker** scoring candidates from an upstream recall stage.
  - The difference is usually organizational/pipeline-boundary labeling rather than a strict architectural disagreement.

- **No direct contradictions on core rationale:** both sources agree the pipeline exists to balance efficiency/latency vs effectiveness/quality.

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
- [[two_tower_model