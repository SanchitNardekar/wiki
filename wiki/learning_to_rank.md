---
slug: learning_to_rank
sources:
- hav4ik.github.io
tags: []
title: Learning to Rank
updated: '2026-04-14'
---

# Learning to Rank (LTR)

**Learning to Rank (LTR)** is a family of [[machine_learning]] methods used in [[information_retrieval]] systems—especially [[web_search]] and [[recommender_systems]]—to produce an ordered list of items (documents, products, videos, etc.) for a user. In production search engines, LTR is typically used in the *ranking / re-ranking* stages to sort a candidate set of retrieved items by predicted relevance (and sometimes by user preferences or business objectives).

This page integrates material from a detailed web-search-oriented overview by Chan Kha Vu (“Learning to Rank in Web Search”) and organizes it into a wiki-style reference.

---

## How LTR fits into a modern search engine stack

Modern search engines can be abstracted into:

- **Indexing (offline, continuous)**
  - Extract features/signals from crawled documents and store them in one or more indexes.
- **Retrieval (online)**
  - Given a query, retrieve **top-*k*** candidate documents (sometimes called **Level-0 ranking** or **matching**).
- **Ranking / Re-ranking (online)**
  - Apply LTR models (or rules/heuristics for simpler systems) to order retrieved candidates by relevance.

### Common index types (web-scale)

- **Inverted index (posting list)**
  - Supports term→document lookup and classic term-based scoring like **TF‑IDF** and **BM25**.
- **Vector index**
  - Stores learned **embeddings** for documents (and enables embedding-based query retrieval).
  - Often built from modern contrastive learning models (e.g., BERT-like encoders for text; vision-text models such as SigLIP for visual search were mentioned as an example).
- **Feature index**
  - Stores many engineered signals and compressed neural features used by later-stage rankers.

### Retrieval approaches

- **Keyword / entity matching** using an inverted index.
- **Embedding-based retrieval**
  - Compute a query embedding; retrieve nearest neighbor document embeddings (cosine or Euclidean similarity).
- **Hybrid retrieval** (common at web scale)
  - Combine keyword matching + embeddings (and sometimes knowledge-graph-based query expansion).

**Scalability note (from source):**
- Exact metric trees (e.g., k‑d trees) are typically *not* used at web scale due to computational/memory constraints; **Approximate Nearest Neighbor (ANN)** search is used to get close to $O(1)$ retrieval time.

---

## Problem formulation

Given:

- a query $ \mathbf{q} $
- a set of $n$ retrieved documents $ \mathcal{D} = \{d_1, \dots, d_n\} $

we want to learn a ranking function $f(\mathbf{q}, \mathcal{D})$ that returns an ordering where the top items are most relevant. Often the model computes a **score** for each document (given query-dependent features), and sorting by score yields the ranking.

Common notation (as used in the source’s supervised-LTR section):

- $ \mathbf{x}_d $: feature vector for a query–document pair $(\mathbf{q}, d)$
- $ f_\theta(\cdot) $: model with parameters $\theta$ (neural net or gradient-boosted trees)
- $ s_i = f_\theta(\mathbf{x}_i) $: predicted score for document $d_i$
- $ d_i \rhd d_j $: document $d_i$ should rank above $d_j$

---

## What is “relevance”?

In practice, “relevance” is not a single signal; it is often treated as a combination of multiple measurement methods:

- **Human labels (offline)**
  - Human judges assign graded relevance (e.g., 1–5) using detailed guidelines (Google and Bing have such processes).
  - Pairwise judgments (“is result A more relevant than result B?”) are also used.
- **Click-through rate (CTR) (implicit feedback)**
  - Cheap but biased (e.g., users click higher-ranked results more often even if less relevant—see [[position_bias]]).
- **Conversion / downstream success**
  - Depends on domain: purchases, revenue, subscriptions, engagement, etc.
  - For e-commerce, an example definition is: buys / searches.

---

## Flavors of Learning to Rank

The source divides LTR into **offline** and **online** methods:

### Offline LTR

Trained on a fixed dataset.

- **Supervised LTR**
  - Uses human-judged labels.
  - Subtypes by loss construction:
    - **Pointwise**: per-document loss
    - **Pairwise**: per-pair loss (learn preferences)
    - **Listwise**: loss over a whole ranked list
- **Counterfactual LTR** (a major part of *unbiased LTR*)
  - Learns from historical interaction logs (clicks, conversions, etc.)
  - Must correct for biases in logged data (see [[counterfactual_learning_to_rank]] if available; otherwise this section serves as the seed).

### Online LTR

- Learns from user interactions *in real time* and updates after interactions.
- Online and counterfactual approaches are “trickier” because they learn from **biased signals**, motivating **Unbiased Learning to Rank** (also called *unbiased LTR*).

**Cross-reference:** This area overlaps heavily with [[bandits]] / [[contextual_bandits]] and [[reinforcement_learning]] in how policies are evaluated and improved, though the source uses LTR-specific terminology (“policy”, “evaluation policy”).

---

## Ranking metrics (offline evaluation)

Common information retrieval metrics mentioned:

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **ERR** (Expected Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain) — described as the most commonly used for graded relevance

### Notes on metric choice

- MAP and MRR are widely used in retrieval, but (per the source) *less used* for web-search ranking with **graded** relevance because they do not directly incorporate graded labels unless modified.
- NDCG explicitly uses graded relevance and rank discounting.

### DCG and NDCG

For truncation level $T$ and label $l_i$ at rank $i$:

$$
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log(1+i)}
$$

$$
NDCG@T = \frac{DCG@T}{\max DCG@T}
$$
where $\max DCG@T$ is the DCG of the ideally sorted list, ensuring $NDCG@T \in [0,1]$.

### ERR (Expected Reciprocal Rank)

ERR models a user scanning results from top to bottom until they find a satisfactory result:

$$
ERR = \sum_{r=1}^n \frac{1}{r} R_r \prod_{i=1}^{r-1}(1-R_i)
,\quad
R_i = \frac{2^{l_i}-1}{2^{l_m}}
$$
where $l_m$ is the maximum label level.

---

## Supervised Learning to Rank (classic methods)

A historically important sequence of work is credited (in the source) to **Christopher Burges** (Microsoft Research):

- **RankNet** (Burges et al., 2005)
- **LambdaRank** and **LambdaMART** (Burges et al., 2006)
- A survey framing: Burges (2010)

The source claims LambdaMART remains a strong baseline and can outperform newer methods on some benchmarks.

### Pairwise approach: RankNet

RankNet models pairwise preference probability using a sigmoid over score differences:

$$
P_{ij} = P(d_i \rhd d_j) = \frac{1}{1 + e^{-\sigma(s_i - s_j)}}
$$

Given a “target” probability $\widetilde{P}_{ij}$ (e.g., from judge agreement), RankNet uses cross-entropy:

$$
\mathcal{L}_{\text{RankNet}}(s_i,s_j) =
-\widetilde{P}_{ij}\log P_{ij}
-(1-\widetilde{P}_{ij})\log(1-P_{ij})
$$

**Historical note (from source):**
- The RankNet paper received the ICML Test of Time Best Paper Award (in 2015).

### Listwise approach: ListNet (Cao et al., 2007)

ListNet is presented as a listwise approach using a Plackett–Luce style distribution:

$$
P_\theta(d_i \mid \mathcal{D}^\mathbf{q}) =
\frac{\exp[f_\theta(d_i^\mathbf{q})]}{\sum_{j=1}^n \exp[f_\theta(d_j^\mathbf{q})]}
$$

Probability of sampling a permutation $\pi$ is expressed as a product over sequential choices from the remaining set. The ListNet loss is defined as cross-entropy between the label-induced distribution and the model-induced distribution over documents.

### LambdaRank and LambdaMART (optimizing metric-sensitive objectives)

**Motivation:** Metrics like NDCG/ERR are not directly differentiable because ranking requires sorting.

- LambdaRank modifies the gradients (“lambdas”) so that pairwise updates are weighted by the **change in the target metric** caused by swapping two items.

For NDCG:

$$
\lambda_{ij} \equiv \frac{\partial \mathcal{C}}{\partial s_i}\cdot |\Delta NDCG_{ij}|
$$

with:

$$
\Delta NDCG_{ij} =
\frac{2^{l_j} - 2^{l_i}}{\max DCG@T}
\left(
\frac{1}{\log(1+i)} - \frac{1}{\log(1+j)}
\right)
$$

**Complexity note (from source):**
- Computing $\Delta NDCG$ for all pairs is $O(n^2)$.
- Naive $\Delta ERR$ across all pairs can be $O(n^3)$, but can be reduced to $O(n^2)$ via tricks described in Burges (2010).

**LambdaMART:**
- Uses gradient-boosted decision trees (“MART”: Multiple Additive Regression Trees) rather than neural networks.
- Often implemented via libraries like RankLib and Microsoft’s **LightGBM**.

---

## Practical example: training LambdaMART with LightGBM (MSLR-WEB30K)

The source walks through training a LambdaMART model using LightGBM on **MSLR‑WEB30K** (Microsoft, 2010):

- ~3.7M documents grouped into ~30k queries
- 136-dimensional feature vectors
- relevance labels from 0 (irrelevant) to 4 (perfectly relevant)
- Dataset is described as a “retired commercial labeling set” from Bing

Key implementation details:

- LightGBM needs **query group sizes** (number of documents per query) to compute ranking losses.
- Example parameters include:
  - `objective: "lambdarank"`
  - `metric: "ndcg"`
  - `ndcg_eval_at: [1, 3, 5, 10]`
  - tree parameters like `num_leaves`, and learning rate.

### Feature importance observations (from that experiment)

When inspecting feature importance:

- High importance by “split count” included:
  - **PageRank** features (PageRank and Site-level PageRank)
  - URL structure features (URL length; number of slashes)
  - “QualityScore” features (from a page quality classifier)
  - BM25-related features (Title BM25, Whole-document BM25)
- High importance by “gain” notably included:
  - **Query–URL click count**, suggesting click frequency is a strong relevance indicator.

**Potential tension / contradiction to watch:**
- The supervised setup assumes *human-labeled relevance*, but the feature set includes **click count** as a feature—an implicit feedback signal which is known to be biased. This is not necessarily a contradiction (click features can still help), but it creates a conceptual mismatch with the later “unbiased LTR” section that emphasizes bias in clicks. In practice, supervised LTR pipelines often use click-derived features carefully (e.g., with debiasing or smoothing), but the source excerpt does not specify safeguards.

---

## Theoretical notes: “Is LambdaRank optimizing a loss?”

The source highlights a long-standing question: whether LambdaRank corresponds to optimizing a well-defined global loss function.

- Donmez et al. (2009) reportedly provided empirical evidence of local optimality using a one-sided Monte Carlo test (sampling random directions and verifying metric decreases).
- Burges et al. (2006) discussed conditions (via Poincaré lemma / exact differentials) under which a global potential function could exist, but the source states that existence “remains unknown across iterations” because sorting and recomputing lambdas depends on current scores.
- Wang et al. (2019) introduced a probabilistic framework (LambdaLoss) that interprets LambdaRank as an EM-like procedure optimizing an upper bound of a defined objective.

---

## LambdaLoss framework (Wang et al., 2019)

**Idea:** Treat the ranked list/permutation $\pi$ as a latent variable and define likelihood of observing labels $ \mathbf{y} $ given scores $ \mathbf{s} $ by marginalizing over permutations:

$$
P(\mathbf{y}\mid \mathbf{s}) = \sum_{\pi\in \Pi} P(\mathbf{y}\mid \mathbf{s},\pi)P(\pi\mid \mathbf{s})
$$

Loss is negative log-likelihood:

$$
\mathcal{L}(\mathbf{y},\mathbf{s}) = -\log \sum_{\pi\in\Pi} P(\mathbf{y}\mid \mathbf{s},\pi)P(\pi\mid \mathbf{s})
$$

The source explains this as analogous to classification: optimize differentiable log-likelihood rather than a non-differentiable evaluation metric.

### Recovering RankNet and LambdaRank-style objectives

- If $P(\mathbf{y}\mid \mathbf{s},\pi)$ is defined via a Bradley–Terry model independent of $\pi$, the objective reduces to a RankNet-like pairwise logistic loss.
- To incorporate rank sensitivity (e.g., NDCG-like gain/discount), the source gives a modified pairwise likelihood exponent weighted by gain and discount differences.
- A distribution over rankings $P(\pi\mid \mathbf{s})$ can be derived by adding noise to scores (Taylor et al., 2008); the LambdaLoss paper uses a **hard assignment** (limit as noise → 0) to reduce computational cost.

---

## Unbiased Learning to Rank (learning from user behavior)

Human labeling is expensive and slow. A natural alternative is to use **click logs** and other implicit feedback. However, clicks are biased observations of relevance.

The source structures this section using (and crediting) a lecture by Oosterhuis et al. (2020).

### Setup (click-based learning assumptions)

- It can be more appropriate to talk about a **query instance** $\mathcal{q}$ that includes user context, not only a raw query string.
- Relevance labels are often treated as **binary** in click models (relevant or not), though graded relevance exists in human labeling.

Notations used in the source:

- $\mathcal{q}$: user query instance (query + context)
- $\pi_\theta^{\mathcal{q}}$: ranking produced by model $f_\theta$
- $\pi_\theta^{\mathcal{q}}(d)$: rank position of document $d$
- $y_d^{\mathcal{q}}$: true (unobserved) relevance
- $o_d^{\mathcal{q}}$: whether relevance was observed (examination/observation indicator)
- $c_d^{\mathcal{q}}$: click indicator
- $\Delta(\mathbf{y}^{\mathcal{q}}, \pi_\theta^{\mathcal{q}})$: a linearly decomposable IR metric (NDCG, MRR, MAP, etc.)
- $\mu(r)$: rank weighting function inside $\Delta$

### Click signal biases

The source lists common biases in implicit feedback:

- **Position bias** ([[position_bias]])
  - Users are more likely to examine/click top-ranked results.
  - Eye-tracking examples (2004 vs 2014) show UI design changes can “flatten” attention distribution.
  - **Operational takeaway:** change in SERP design can change bias; estimators should be recalibrated after major design changes.
- **Selection bias**
  - Some items have *zero probability* of being examined (e.g., users rarely go to page 2+).
  - Important distinction from position bias: some correction methods assume every item has non-zero examination probability.
- **Trust bias**
  - Users may click top items because they trust the ranking system, not necessarily because those results are relevant.

---

## Counterfactual Learning to Rank (high-level)

The source introduces counterfactual LTR as learning/evaluating from historical interaction logs.

Core idea:

- **Counterfactual evaluation:** evaluate a new ranking function $f_\theta$ using interaction data collected under a previously deployed ranking function $f_{\text{deploy}}$.

Terminology mapping (from the unbiased LTR literature):

- $f_{\text{deploy}}$: **behavior policy** (a “policy” that generated the logged data)
- $f_\theta$: **evaluation policy** (the candidate ranker being evaluated)

> **Note:** The provided source text cuts off mid-sentence during this section, so details like IPS estimators, click models (PBM/DBN), intervention/randomization strategies, and typical counterfactual objectives are not included in the excerpt. This page should be extended once additional sources are provided.

---

## Notes, scope, and limitations (from the source)

- The referenced write-up is explicitly “biased towards Web Search” and “not exhaustive.”
- The author states they are “far from an expert” and that the post may contain inaccuracies.
- Information is described as coming from published papers/public sources; no proprietary implementation details are intended.

---

## Related pages (suggested)

Add or link to these pages where they exist:

- [[information_retrieval]]
- [[web_search]]
- [[recommender_systems]]
- [[bm25]]
- [[tf_idf]]
- [[embeddings]]
- [[approximate_nearest_neighbors]]
- [[learning_to_rank_metrics]] (if metrics are split out later)
- [[position_bias]]
- [[counterfactual_learning]]
- [[contextual_bandits]]
- [[lightgbm]]
- [[gradient_boosted_decision_trees]]

---

## Contradictions and tensions to track

Since this is a new page with a single main source, there are no contradictions with prior page content. However, within the integrated material there are some **