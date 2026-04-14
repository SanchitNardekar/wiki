---
slug: loss_functions_and_optimization
sources:
- hav4ik.github.io
- blog.reachsumit.com
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

### Extension: multi-stage/cascade ranking and where losses change by stage

The new source emphasizes **cascade / multi-stage ranking systems** driven by **latency constraints**, where different stages optimize different objectives:

- Earlier stages (retrieval/recall and **pre-ranking**) prioritize:
  - **efficiency** (scoring many candidates quickly)
  - **recall-oriented metrics** (keep good candidates in the top set passed downstream)
- Later stages (ranking and re-ranking) can afford:
  - more expensive models (e.g., cross-feature interaction models)
  - losses aligned with downstream ranking metrics (often [[ndcg]]@k)

This complements (and extends) the earlier “retrieval vs re-ranking” framing: in many industrial stacks there is an explicit **pre-ranking** stage between retrieval and final ranking.

Cross-references: [[information_retrieval]], [[learning_to_rank]].

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

## Extension: losses and optimization for Two-Tower / Dual-Encoder pre-ranking

The new source focuses on **two-tower (dual-encoder / bi-encoder)** models as a common choice for **pre-ranking** (and sometimes retrieval) because they separate representation learning and scoring:

- Compute a query/user embedding \(u\) and a document/item embedding \(v\) **independently**
- Score with a cheap similarity, often **inner product**:
  - \(s(u,v) = u^\top v\)

This “late interaction”/decoupled design enables:
- precomputing and indexing document/item embeddings (fast inference)
- approximate nearest neighbor search over embeddings (fast candidate generation)
- serving updates by adding/updating item embeddings without retraining the whole model (for some workflows)

Cross-references: [[information_retrieval]], [[learning_to_rank]].

### Two-tower vs. interaction-heavy rankers: implication for objectives

The source contrasts architecture families (in neural matching / retrieval terms):

- **Representation-based**: two-tower / dual encoder (late interaction at the end)
- **Interaction-based**: DRMM/KNRM-style interaction matrices + neural scorer; **cross-encoders** like BERT
- **Contextual late interaction**: ColBERT-style token-level interactions while keeping query/document decoupling

**Optimization implication** (high-level):
- Two-tower models typically optimize a **similarity learning objective** in embedding space (often using negative sampling / contrastive losses), rather than a listwise NDCG surrogate directly.
- Final re-rankers (GBDT or cross-encoders) more often optimize metric-aligned surrogates (e.g., LambdaRank-style).

Cross-references: [[ndcg]], [[ranking_metrics]].

### Dual encoder variants and a reported empirical finding

The source describes **Siamese Dual Encoder (SDE)** vs **Asymmetric Dual Encoder (ADE)**:

- **SDE (Siamese)**: two towers share parameters (or are identical subnetworks)
- **ADE (Asymmetric)**: towers have distinct parameters

Reported conclusion (Dong et al., 2022, per source summary):
- SDEs performed **better** than ADEs on a QA retrieval task, attributed to ADEs embedding inputs into **disjoint embedding spaces** which can hurt retrieval quality.
- ADE performance can be improved by **sharing a projection layer** (ADE-SPL), making it competitive with (or better than) SDE.
- Sharing/freezeing token embedders (ADE-STE, ADE-FTE) yields only marginal improvements.

> Note: This is task- and setup-dependent; treat it as an empirical observation reported in the source, not a universal law.

### Interaction-enhanced two-tower models add auxiliary losses (contrastive + logloss)

The source highlights a common limitation of pure two-tower models:
- **Lack of information interaction between towers** (because embeddings are learned mostly independently)

Proposed extensions introduce additional interaction modules and **additional losses**.

#### IntTower: combining supervised classification loss + contrastive regularization

The source describes an “Interaction Enhanced Two Tower Model (IntTower)” which combines:
- Feature refinement (Light-SE block inspired by SENet)
- Early/fine-grained feature interactions (FE-block inspired by ColBERT’s interaction style)
- **Contrastive Interaction Regularization (CIR)** using **InfoNCE** loss

Training objective (as described at a high level in the source):
- combine **logloss** (binary cross-entropy on predicted score vs label) **+** an **InfoNCE** contrastive loss that pulls user/query closer to positive items than negatives

Cross-references:
- [[contrastive_learning]] (for InfoNCE-style losses)
- [[cross_entropy]] (for “logloss” / BCE)
- [[information_retrieval]] (for pre-ranking vs ranking)

##### InfoNCE (conceptual form)

While the source does not give a full equation, the commonly used InfoNCE form for an anchor \(u\), positive \(v^+\), and negatives \(\{v_k^-\}\) is:

\[
\mathcal{L}_{\text{InfoNCE}}(u, v^+) =
- \log \frac{\exp(\text{sim}(u, v^+)/\tau)}
{\exp(\text{sim}(u, v^+)/\tau) + \sum_k \exp(\text{sim}(u, v_k^-)/\tau)}
\]

- \(\text{sim}\) is often dot product or cosine similarity
- \(\tau\) is a temperature parameter

**How this connects to ranking**:
- InfoNCE encourages **relative ordering** (positive above negatives) in embedding similarity space, which indirectly improves retrieval/pre-ranking quality.
- This is conceptually aligned with pairwise ranking (positives vs negatives), but implemented as a multi-negative softmax.

---

## Deep Metric Learning losses as ranking-friendly objectives (Contrastive/Triplet/Margin-Softmax)

The new source (“Deep Metric Learning: a (Long) Survey”, Chan Kha Vu, 2021) adds a complementary view: many retrieval / pre-ranking systems can be trained using **deep metric learning (DML)** objectives that directly shape the embedding geometry.

This section does **not replace** LTR listwise methods (e.g., LambdaMART). Instead, it explains a commonly used objective family for **embedding-based retrieval and pre-ranking** in [[information_retrieval]] stacks.

Cross-references:
- [[contrastive_learning]] (general contrastive family; InfoNCE and beyond)
- [[learning_to_rank]] (where these fit relative to pairwise/listwise LTR)

### Problem setting (supervised metric learning)

Given labeled samples \(x \in \mathcal{X}\) with discrete labels \(y \in \mathcal{Y}\), train an embedding model:

\[
f_{\theta}(\cdot): \mathcal{X} \to \mathbb{R}^n
\]

with a (usually fixed) distance function \(\mathcal{D}\) such that:

- \(\mathcal{D}(f_\theta(x_1), f_\theta(x_2))\) is **small** if \(y_1=y_2\)
- and **large** otherwise.

Common choices:
- Euclidean distance \(\|p-q\|_2\)
- cosine distance / similarity (especially with normalized embeddings)

**Connection to ranking:** a retrieval system ranks candidates by a similarity score \(s(u,v)\) (dot/cosine); DML losses make this similarity **order positives above negatives**, aligning with ranking needs at retrieval/pre-ranking time.

### “Direct” contrastive approaches (pair/triplet-based)

These objectives are “direct” in the sense that they explicitly pull positives together and push negatives apart.

#### Contrastive loss (pair-based)

For two samples \((x_1,y_1)\), \((x_2,y_2)\) with a margin \(\alpha\):

\[
\mathcal{L}_\text{contrast} =
\mathbb{1}_{y_1 = y_2} \, \mathcal{D}^2_{f_\theta}(x_1, x_2)
+
\mathbb{1}_{y_1 \ne y_2} \, \max(0, \alpha - \mathcal{D}^2_{f_\theta}(x_1, x_2))
\]

- The **margin** prevents collapse to a single point embedding.

#### Triplet loss (anchor/positive/negative)

For anchor \(x_a\), positive \(x_p\) (same class), negative \(x_n\) (different