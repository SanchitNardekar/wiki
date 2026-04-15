---
slug: loss_functions_and_optimization
sources:
- hav4ik.github.io
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
- blog.ezyang.com
tags: []
title: Loss Functions and Optimization
updated: '2026-04-15'
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

$$
DCG@T = \sum_{i=1}^{T} \frac{2^{l_i}-1}{\log(1+i)}
$$

$$
NDCG@T = \frac{DCG@T}{\max DCG@T}
$$

- **lᵢ** is the relevance label at rank i (graded).
- NDCG is in **[0, 1]** due to normalization.

### ERR recap (user stops when satisfied)

$$
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} (1-R_i), \quad
R_i = \frac{2^{l_i}-1}{2^{l_m}}
$$

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

$$
P_{ij} = P(d_i \rhd d_j) = \frac{1}{1 + e^{-\sigma (s_i - s_j)}}
$$

- **σ** controls sigmoid steepness.

Let **\~Pᵢⱼ** be the target probability that i should rank above j (e.g., from human preference aggregation).

### RankNet cross-entropy loss

$$
\mathcal{L}_{\text{RankNet}}(s_i, s_j) =
- \widetilde{P}_{ij}\log P_{ij} - (1-\widetilde{P}_{ij})\log(1-P_{ij})
$$

Properties highlighted:

- The cost is **symmetric**: swapping (i, j) and flipping probabilities leaves it unchanged.
- Training sums over a set of preferred pairs.

**Impact note:** RankNet is described as a precursor to LambdaRank/LambdaMART and won an ICML Test of Time award (2015) per the source.

---

## Listwise optimization: ListNet loss (Cao et al., 2007)

ListNet is a listwise approach that defines a distribution over documents using a Plackett–Luce-style model and uses cross entropy between distributions derived from labels and model scores.

### Top-1 Plackett–Luce probability

$$
P_\theta(d_i^q \mid \mathcal{D}^q) =
\frac{\exp[f_\theta(d_i^q)]}{\sum_{j=1}^n \exp[f_\theta(d_j^q)]}
$$

### ListNet cross-entropy loss (top-1 version shown in the source)

$$
\mathcal{L}_{\text{ListNet}}(s^q, y^q)
= -\sum_{i=1}^n
\frac{\exp[y_i^q]}{\sum_{j=1}^n \exp[y_j^q]}
\log \left(
\frac{\exp[f_\theta(d_i^q)]}{\sum_{j=1}^n \exp[f_\theta(d_j^q)]}
\right)
$$

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

Start from the RankNet gradient term $\lambda_{ij} = \partial \mathcal{C}/\partial s_i$ and then reweight it by the **absolute change** in a target metric if i and j swap positions.

For NDCG:

$$
\lambda_{ij} \equiv \frac{\partial \mathcal{C}}{\partial s_i}\cdot |\Delta NDCG_{ij}|
$$

Where:

$$
\Delta NDCG_{ij} =
\frac{2^{l_j}-2^{l_i}}{\max DCG@T}
\left(
\frac{1}{\log(1+i)}-\frac{1}{\log(1+j)}
\right)
$$

Computational note (from the source):

- Computing $\Delta NDCG_{ij}$ for all pairs is **O(n²)**.
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

- Compute a query/user embedding $u$ and a document/item embedding $v$ **independently**
- Score with a cheap similarity, often **inner product**:
  - $s(u,v) = u^\top v$

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

---

## Extension: controlling precision in embedding-based retrieval via relevance filtering (Cosine Adapter)

**New source (Rossi et al., CIKM 2024)** emphasizes a practical issue in embedding-based retrieval and pre-ranking: dense retrieval systems often maximize **recall**, but **low precision** can harm UX—especially in **product search** where queries may have **few truly relevant items**.

Key points:

- ANN-based dense retrieval has **no natural cutoff** analogous to lexical keyword matching; you always get top‑K neighbors even if most are irrelevant.
- Raw **cosine similarity** values are often:
  - **hard to interpret** as absolute relevance, and
  - **not comparable across queries**, because dual encoders are typically trained to get correct *relative ordering* within a query context, not calibrated absolute scores.
- Relying on:
  - a fixed **top‑K**, or
  - a global **cosine similarity threshold**
  is often insufficient for filtering irrelevant results.

Cross-references: [[information_retrieval]], [[learning_to_rank]], [[ranking_metrics]].

### Contradiction / tension with some common practice

- **Common heuristic**: “Apply a global cosine threshold to dense retrieval scores.”
- **New source claim**: cosine scores “should not be compared across different queries,” so a global threshold on raw cosine is generally **suboptimal**.

This does not strictly contradict the existing page (which did not claim cosine is calibrated), but it **explicitly challenges** a widely used operational shortcut.

### Position in a cascade: retrieval → (filter) → re-ranking

Rossi et al. propose inserting a lightweight **relevance filtering** step after ANN retrieval and before re-ranking:

1. Dual encoder produces query embedding and document embeddings; ANN retrieves top‑K with cosine scores.
2. **Cosine Adapter** maps cosine scores into **interpretable, query-comparable relevance probabilities/logits**.
3. Apply a **single global threshold** on calibrated score to filter candidates.
4. Forward filtered set to downstream reranker.

This can reduce wasted reranker computation on obviously irrelevant candidates.

### Why interpretability/calibration matters

The source highlights a specific reason calibration is needed:

- Contrastive and listwise objectives used for dual encoders shape embedding space for **relative distances**; thus cosine values act like **ranking scores**, not **probabilities**.
- As a result, the same cosine value may indicate “high relevance” for one query but “low relevance” for another query (query difficulty / ambiguity, etc.).

### Dual-encoder training losses referenced (contrastive vs listwise softmax)

The paper restates two common dual-encoder objectives (useful for this page because they connect optimization choices to score calibration issues):

#### In-batch contrastive / InfoNCE-style loss (per query $q_i$)

$$
\text{loss}_i = - \log \frac{\exp(\cos(q_i,p_i)/\tau)}
{\exp(\cos(q_i,p_i)/\tau) + \sum_{j\in N}\exp(\cos(q_i,p_j)/\tau)}
$$

- $p_i$ is the positive passage/product for query $q_i$
- $N$ are in-batch negatives
- $\tau$ is temperature

Cross-reference: [[contrastive_learning]].

#### Softmax listwise loss over candidate set $P_i$

$$
\text{loss}_i = -\sum_{j\in P_i} y_{ij}\;
\log\frac{\exp(\cos(q_i,p_j)/\tau)}{\sum_{j\in P_i}\exp(\cos(q_i,p_j)/\tau)}
$$

- $y_{ij}$ are predefined labels (can be graded)
- Still fundamentally shapes **relative** scores, not absolute calibration

**Interpretation in this wiki’s framing:** These are listwise/contrastive objectives for *embedding retrieval*, but they are not directly optimizing a rank metric like [[ndcg]]—and they do not inherently yield **cross-query comparable scores**.

### Cosine Adapter: query-dependent monotonic calibration of cosine similarity

The **Cosine Adapter** is a small neural module that takes the **query embedding** as input and outputs parameters $\Theta$ for a **monotonic transformation** $F_\Theta(\cdot)$ that maps cosine similarity $x \in [-1,1]$ to a calibrated logit.

Filtering is then:

$$
\tilde{P}_i = \{p_j \mid F_\Theta(\cos(q_i,p_j)) \ge t\}
$$

- $\Theta$ is **query-dependent** (computed once per query)
- $t$ is a **global threshold** tuned offline

**Important design constraint:** $F$ is chosen monotonic to preserve the ANN ranking order as much as possible (minimizing recall impact).

The paper explores several mapping families (baseline plus parameterized shapes):

- Raw: $F(x)=x$
- Linear: $F(x\mid a,b)=ax$

> Note: the existing page content truncates the Linear mapping (it likely intended $ax+b$).