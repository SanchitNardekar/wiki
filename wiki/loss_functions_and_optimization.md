---
slug: loss_functions_and_optimization
sources:
- hav4ik.github.io
tags: []
title: Loss Functions and Optimization
updated: '2026-04-14'
---

# Loss Functions and Optimization

This page summarizes common **loss functions** and **optimization strategies** for **Learning to Rank (LTR)** in web search and recommendation-style ranking systems. It focuses on the practical tension that drives many LTR losses:

- The metrics we care about (e.g., [[ndcg]], ERR) depend on **sorting**, which is **non-differentiable**.
- Therefore, we optimize **surrogate objectives** (pointwise/pairwise/listwise) or use frameworks that create useful **gradients (“lambdas”)** aligned with ranking metrics.

Related pages to cross-reference:
- [[learning_to_rank]] (overview of LTR problem setting and families of methods)
- [[ranking_metrics]] (MAP, MRR, NDCG, ERR definitions and use cases)
- [[unbiased_learning_to_rank]] (biases in click logs and counterfactual objectives)
- [[lightgbm]] (practical LambdaMART training)
- [[information_retrieval]] (retrieval vs re-ranking stack)

---

## Context: where ranking losses fit in a search system

A modern search system is often described as a pipeline:

- **Indexing (offline)**: build multiple indices and store signals/features.
  - Inverted index (posting lists) for term matching and TF-IDF/BM25 scoring
  - Vector index for dense embeddings (e.g., BERT-like text embeddings; visual embeddings such as SigLIP were mentioned as an example)
  - Feature index: thousands of engineered and learned signals used in later stages
- **Top‑k retrieval / matching (“Level‑0 ranking”)**:
  - keyword/entity matching + embedding-based retrieval (kNN by cosine/Euclidean similarity)
  - Approximate Nearest Neighbors (ANN) is preferred at large scale (rather than metric trees like k-d trees) to approach ~O(1) retrieval behavior
- **Re-ranking (Learning to Rank)**:
  - ML-based ranking (classification/regression → scoring → sort) dominates at large-scale web search engines
  - Rule-based heuristics may be sufficient for smaller systems

Loss functions and optimization methods in this page primarily apply to the **re-ranking** stage, where rich features are available.

---

## Learning-to-Rank problem formulation (scoring + sorting)

Given a query **q** and retrieved documents **D = {d₁, …, dₙ}**, learn a function:

- **f(q, D)** or more commonly a scoring function **sᵢ = fθ(xᵢ)** for each document
- Rank documents by sorting scores **sᵢ**

Common notation used in ranking losses:

- **xᵢ**: feature vector for query-document pair (q, dᵢ)
- **sᵢ**: model score for dᵢ
- **dᵢ ≻ dⱼ**: document i should rank higher than j (preference)

---

## Ranking metrics and why they are hard to optimize directly

Common ranking quality metrics include:

- **MAP**, **MRR**: widely used for retrieval; often assume binary relevance
- **NDCG@T**: the most common graded relevance metric for ranking
- **ERR**: models user scanning down the list until satisfaction

### NDCG recap (graded relevance; position-sensitive)

For truncation level **T**:

\[
DCG@T = \sum_{i=1}^{T} \frac{2^{l_i}-1}{\log(1+i)}
\]

\[
NDCG@T = \frac{DCG@T}{\max DCG@T}
\]

- **lᵢ** is the relevance label at rank i (graded).
- NDCG is in **[0, 1]** due to normalization.

### ERR recap (user stops when satisfied)

\[
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} (1-R_i), \quad
R_i = \frac{2^{l_i}-1}{2^{l_m}}
\]

### Key optimization issue

- Metrics like NDCG/ERR require **sorting**, which makes them **non-differentiable** and often **flat/discontinuous** w.r.t. model parameters.
- LTR training therefore relies on **surrogates** or **gradient shaping**.

---

## Families of supervised ranking losses

Supervised LTR methods are commonly grouped by how the loss is constructed:

- **Pointwise**: treats each document independently (regression/classification on relevance)
- **Pairwise**: compares document pairs (preference learning)
- **Listwise**: uses the full list distribution or list-level objective

(These groupings are widely used; the source text emphasized pointwise/pairwise/listwise as the “flavors” of supervised LTR.)

---

## Pairwise optimization: RankNet loss (Burges et al., 2005)

RankNet reframes ranking as a pairwise probabilistic preference learning problem solvable by gradient descent.

### Pairwise probability model

\[
P_{ij} = P(d_i \rhd d_j) = \frac{1}{1 + e^{-\sigma (s_i - s_j)}}
\]

- **σ** controls sigmoid steepness.

Let **\~Pᵢⱼ** be the target probability that i should rank above j (e.g., from human preference aggregation).

### RankNet cross-entropy loss

\[
\mathcal{L}_{\text{RankNet}}(s_i, s_j) =
- \widetilde{P}_{ij}\log P_{ij} - (1-\widetilde{P}_{ij})\log(1-P_{ij})
\]

Properties highlighted:

- The cost is **symmetric**: swapping (i, j) and flipping probabilities leaves it unchanged.
- Training sums over a set of preferred pairs.

**Impact note:** RankNet is described as a precursor to LambdaRank/LambdaMART and won an ICML Test of Time award (2015) per the source.

---

## Listwise optimization: ListNet loss (Cao et al., 2007)

ListNet is a listwise approach that defines a distribution over documents using a Plackett–Luce-style model and uses cross entropy between distributions derived from labels and model scores.

### Top-1 Plackett–Luce probability

\[
P_\theta(d_i^q \mid \mathcal{D}^q) =
\frac{\exp[f_\theta(d_i^q)]}{\sum_{j=1}^n \exp[f_\theta(d_j^q)]}
\]

### ListNet cross-entropy loss (top-1 version shown in the source)

\[
\mathcal{L}_{\text{ListNet}}(s^q, y^q)
= -\sum_{i=1}^n
\frac{\exp[y_i^q]}{\sum_{j=1}^n \exp[y_j^q]}
\log \left(
\frac{\exp[f_\theta(d_i^q)]}{\sum_{j=1}^n \exp[f_\theta(d_j^q)]}
\right)
\]

Intuition:

- Convert both **labels** and **scores** into probability distributions over documents.
- Minimize cross entropy between them.

---

## LambdaRank and LambdaMART: metric-aware gradient shaping (Burges et al., 2006)

### Motivation: RankNet optimizes pairwise errors, not position-sensitive metrics

The source explicitly notes a limitation:

- RankNet’s objective is a smooth approximation to **pairwise error count**.
- It does **not** yield desirable gradients for **position-sensitive objectives** like NDCG/ERR.

### Core idea: “lambdas” weighted by metric change

Start from the RankNet gradient term \(\lambda_{ij} = \partial \mathcal{C}/\partial s_i\) and then reweight it by the **absolute change** in a target metric if i and j swap positions.

For NDCG:

\[
\lambda_{ij} \equiv \frac{\partial \mathcal{C}}{\partial s_i}\cdot |\Delta NDCG_{ij}|
\]

Where:

\[
\Delta NDCG_{ij} =
\frac{2^{l_j}-2^{l_i}}{\max DCG@T}
\left(
\frac{1}{\log(1+i)}-\frac{1}{\log(1+j)}
\right)
\]

Computational note (from the source):

- Computing \(\Delta NDCG_{ij}\) for all pairs is **O(n²)**.
- A naive ERR swap-change computation can be **O(n³)**, but there are tricks (Burges, 2010) to reduce to **O(n²)**.

### LambdaMART

- LambdaMART uses the LambdaRank gradient idea but optimizes in **function space** via **gradient-boosted decision trees** (MART = Multiple Additive Regression Trees).
- In practice, it remains a very strong baseline and can outperform newer methods on some benchmarks (as stated in the source).

Cross-reference: [[gradient_boosting]], [[gbdt]], [[lightgbm]].

---

## Practical optimization: training LambdaMART with LightGBM

The source provides a concrete recipe using the **MSLR-WEB30K** dataset (Bing, released 2010):

- ~3.7M documents, grouped into 30k queries
- 136-dimensional feature vectors
- relevance labels in {0, 1, 2, 3, 4}

### LightGBM setup (high-level)

Key requirements:

- Provide **group/query sizes** to LightGBM so it can compute query-level ranking losses/metrics.

Example parameter highlights from the source:

- `objective: "lambdarank"`
- `metric: "ndcg"`
- `ndcg_eval_at: [1, 3, 5, 10]`

### Interpreting trained models

Feature importance can be inspected (split-based and gain-based). Observations reported for this dataset:

- Split-importance leaders included:
  - Site-level PageRank and PageRank (features #131, #130)
  - URL length and number of slashes (#127, #126) as surprisingly strong indicators in that older dataset
  - Page quality classifier outputs (#133, #132)
  - BM25 features (e.g., Title BM25, Whole document BM25)
- Gain-importance top feature included:
  - Query‑URL click count (#134)

This underscores a theme: **features matter** greatly, and training a strong ranker is often as much about **feature construction** as it is about loss choice.

---

## Theoretical status: is LambdaRank optimizing a “real” loss?

The source describes long-standing theoretical questions:

- Does LambdaRank converge?
- Is there a global underlying loss function across iterations?

Evidence and developments summarized:

- Donmez et al. (2009): empirical local optimality via one-sided Monte Carlo test (sampling random directions and verifying metric decreases with small steps).
- Burges et al. (2006): attempted justification using Poincaré lemma (exact/closed forms).
  - Symmetry conditions can hold at fixed weights, but global existence across iterations is unclear because lambdas depend on rankings induced by current scores.
- Wang et al. (2019): provides a probabilistic framework (LambdaLoss) showing LambdaRank corresponds to an EM-like optimization of a well-defined objective / upper bound.

---

## LambdaLoss (Wang et al., 2019): a probabilistic framework for metric-driven objectives

LambdaLoss treats the **ranked list** as a latent variable and defines a likelihood of observing labels given scores by mixing over permutations.

### Generic mixture likelihood

Let **s** be scores, **π** a permutation (ranking), and **Π** the set of all permutations:

\[
P(y \mid s) = \sum_{\pi \in \Pi} P(y \mid s, \pi) P(\pi \mid s)
\]

Loss is negative log-likelihood:

\[
\mathcal{L}(y, s) = -\log \sum_{\pi \in \Pi} P(y \mid s, \pi)P(\pi \mid s)
\]

Optimization interpretation:

- Can be optimized via an **EM process**:
  - E-step: compute \(P(\pi \mid s)\) from current scores
  - M-step: update model to minimize negative log-likelihood

### Recovering RankNet in this framework

If preferences are modeled via a Bradley–Terry style likelihood independent of π, the loss reduces to a RankNet-like form (pairwise logistic loss).

### Making it rank-sensitive (toward NDCG-like behavior)

Define gain/discount functions analogous to NDCG:

\[
G(i) = \frac{2^{y_i}-1}{\max DCG}, \quad D(i)=\log(1+\pi_i)
\]

Modify preference likelihood to include rank positions:

\[
P( y_i > y_j \mid s_i, s_j, \pi_i, \pi_j)
=
\left(\frac{1}{1 + e^{-\sigma(s_i - s_j)}}\right)^{
|G(i)-G(j)|\cdot \left|\frac{1}{D(\pi_i)} - \frac{1}{D(\pi_j)}\right|
}
\]

The paper also considers distributions over π induced by noisy scores; the source notes using a **hard assignment** limit (ε → 0) for computational reasons.

**Claim noted in the source:** Wang et al. (2019) prove LambdaRank can be viewed as an EM procedure optimizing this LambdaLoss objective (or an upper bound), and the framework enables designing new metric-driven losses.

Cross-reference: [[expectation_maximization]], [[probabilistic_models]].

---

## Unbiased / counterfactual learning-to-rank: when “labels” come from clicks (preview)

Although this page focuses on loss functions and optimization, the source also motivates why optimization objectives change when training data comes from **implicit feedback**:

- Human labeling is expensive and process-heavy (e.g., long evaluation guidelines at major search engines).
- Clicks are cheaper but biased.

Common click biases mentioned:

- **Position bias**: top-ranked results get examined/clicked more.
- **Selection bias**: items beyond the first page may have near-zero probability of examination.
- **Trust bias**: users may click higher-ranked items because they trust the system, not because items are truly more relevant.

These biases matter because naïvely optimizing a click-based loss may reinforce the deployed ranking’s bias. Methods that correct for these are often called **Unbiased Learning to Rank**.

Cross-reference: [[counterfactual_learning_to_rank]], [[propensity_scoring]], [[position_bias]].

---

## Notes on contradictions / uncertainty in the new source

Since this wiki page is currently seeded from a single blog-style source, the source itself includes an explicit caveat:

- The author states they are “far from an expert” and that the post “likely contains inaccuracies.”

No direct contradictions with existing page content exist (this was a new page). However, keep in mind a potential tension that commonly appears in the literature and is hinted at here:

- **LambdaRank described as “directly optimizing NDCG”** vs. the later discussion that **the global loss was historically unclear** and later justified via LambdaLoss/EM-like interpretations.
  - These statements can be reconciled as: LambdaRank produces **gradients aligned with NDCG improvements** (practically effective), while the existence of a single smooth global objective was historically debated and later formalized under frameworks like LambdaLoss.

---

## Quick “when to use what” (practical guidance)

- Use **RankNet-style pairwise logistic loss** when:
  - pairwise preferences are reliable
  - you want a simple, stable differentiable objective
- Use **ListNet** when:
  - you prefer listwise distribution matching and can train with full query groups
- Use **LambdaMART (LambdaRank + GBDT)** when:
  - you want a high-performing baseline for web-search-style tabular features
  - you care about NDCG@k and want strong out-of-the-box performance (e.g., via [[lightgbm]])
- Consider **LambdaLoss-style objectives** when:
  - you want a principled probabilistic interpretation and metric-driven loss design

For click-based learning, prefer **counterfactual/unbiased objectives** rather than treating clicks as ground truth labels.

---