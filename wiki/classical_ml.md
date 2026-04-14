---
slug: classical_ml
sources:
- hav4ik.github.io
tags: []
title: Classical ML Algorithms
updated: '2026-04-14'
---

# Classical ML Algorithms

This page surveys **classical machine learning algorithms** (non–end-to-end deep neural approaches) and how they appear in real production systems—especially **information retrieval (IR)** and **Learning to Rank (LTR)** used in web search and recommender systems.

> Related pages (see also): [[information_retrieval]], [[ranking]], [[learning_to_rank]], [[gbdt]], [[logistic_regression]], [[counterfactual_learning]], [[recommendation_systems]], [[approximate_nearest_neighbors]]

---

## Where “classical ML” shows up in search & ranking systems

Modern search engines and many recommender systems follow a multi-stage pipeline:

- **Indexing (offline, continuous)**
  - Extract signals/features for items (“documents”) and store them in one or more indexes.
  - Common index types mentioned in the source:
    - **Inverted index / posting lists**: term → documents mapping; supports lexical scores such as **TF‑IDF** and **BM25**.
    - **Vector index**: stores embedding vectors for documents/queries (often produced by neural encoders); used for nearest-neighbor retrieval.
    - **Feature index**: large collection of engineered signals and compressed representations used for final ranking / reranking.

- **Top‑k retrieval (a.k.a. “matching”, “level‑0 ranking”)**
  - Retrieves a candidate set of potentially relevant documents for a query.
  - At web scale, often a **hybrid**:
    - Keyword/entity matching via an inverted index (lexical retrieval; e.g., BM25).
    - Embedding-based retrieval via nearest neighbors in a vector index.
  - **Approximate Nearest Neighbors (ANN)** are commonly used for efficiency; the source notes that metric trees like k‑d trees are typically not used at large scale due to memory/latency concerns.

- **Ranking / reranking (Learning to Rank)**
  - The candidate documents are ordered by predicted relevance (and possibly personalization).
  - For small systems, heuristic or rule-based ranking can be sufficient; for large systems, **ML-based ranking** (LTR) is standard.

---

## Learning to Rank (LTR) as a “classical ML” family

**Learning to Rank** learns a scoring function \(f(\mathbf{q}, d)\) (often written \(f_\theta(\mathbf{x}_{q,d})\)) that assigns a score to each document \(d\) for a query \(\mathbf{q}\); sorting by score gives the ranked list.

- In many systems, LTR is the *final* stage that produces the list shown to the user (closely tied to [[recommendation_systems]] and [[information_retrieval]]).
- LTR models are commonly implemented with:
  - **GBDTs** (e.g., LambdaMART in LightGBM) — see [[gbdt]]
  - or neural nets (RankNet-style objectives) — see also [[deep_learning]] (if present elsewhere)

---

## Relevance signals and labels

Ranking quality depends on how “relevance” is defined/observed. The source describes three common components:

- **Human-labeled relevance**
  - Offline judgments (e.g., graded relevance 1–5, or pairwise preferences).
  - High quality but expensive; requires strict guidelines and trained raters.

- **Click-through rate (CTR)**
  - Cheap implicit signal but biased (notably by rank/position and UI).

- **Conversion rate**
  - Business-defined success (purchases, sales, profit, etc.); in e-commerce often “buys per search”.

---

## Flavors of LTR methods

LTR methods can be categorized along multiple axes:

### Offline vs online LTR
- **Offline LTR**: train from a fixed dataset.
- **Online LTR**: learn from user interactions in real time; model updates after interactions.

### Supervised vs counterfactual (learning from behavior)
- **Supervised LTR**: train on labeled query-document relevance judgments.
- **Counterfactual LTR**: train from historical interaction logs (clicks, conversions), attempting to correct bias.
  - Often grouped under **Unbiased LTR** — see [[counterfactual_learning]].

### Pointwise vs pairwise vs listwise objectives
- **Pointwise**: loss defined per document (treat ranking like regression/classification).
- **Pairwise**: loss defined on pairs of documents (learn preferences).
- **Listwise**: loss defined on entire ranked lists.

> Note: The source explicitly highlights **online and counterfactual** methods as “trickier” because they learn from **biased** user signals; “Unbiased LTR” refers to methods that correct for these biases.

---

## Core ranking metrics (IR evaluation)

Common offline ranking metrics include:

- **MAP (Mean Average Precision)** and **MRR (Mean Reciprocal Rank)**
  - Widely used in retrieval; the source notes they are less used for graded relevance ranking because they don’t directly incorporate graded labels (unless adapted or treated as binary).

- **NDCG (Normalized Discounted Cumulative Gain)** — most common in graded relevance ranking  
  Define truncated DCG:
  \[
  DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log(1+i)}
  \]
  and normalize:
  \[
  NDCG@T = \frac{DCG@T}{\max DCG@T}
  \]
  where \(l_i\) is the graded relevance label at rank \(i\).

- **ERR (Expected Reciprocal Rank)**
  Models a user scanning results top-down until satisfied:
  \[
  ERR = \sum_{r=1}^n \frac{1}{r} R_r \prod_{i=1}^{r-1}(1-R_i), \quad
  R_i = \frac{2^{l_i}-1}{2^{l_m}}
  \]
  where \(l_m\) is the maximum label value.

See also: [[metrics]] (if present), [[information_retrieval]].

---

## Classical supervised LTR algorithms

A historically influential sequence of “classical” LTR methods came from Microsoft Research work (Burges et al.):

### RankNet (pairwise; Burges et al. 2005)
RankNet models the probability that document \(i\) should rank above document \(j\) using a sigmoid on score differences:

\[
P_{ij} = \frac{1}{1+e^{-\sigma (s_i - s_j)}}
\]

Given target preference probability \(\widetilde{P}_{ij}\), optimize cross-entropy:

\[
\mathcal{L}_{\text{RankNet}}(s_i, s_j) =
- \widetilde{P}_{ij}\log P_{ij} - (1-\widetilde{P}_{ij})\log(1-P_{ij})
\]

Connections to classical ML:
- Equivalent in spirit to **pairwise logistic regression** on score differences — see [[logistic_regression]].

### ListNet (listwise; Cao et al. 2007)
ListNet defines a probability distribution over permutations using a Plackett–Luce-style model and minimizes listwise cross-entropy between label-induced and model-induced distributions.

This is a canonical **listwise** alternative to pairwise RankNet/LambdaRank.

### LambdaRank and LambdaMART (Burges et al. 2006)
Problem: metrics like **NDCG/ERR are non-differentiable** because ranking requires sorting.

LambdaRank idea:
- Start from RankNet-like pairwise gradients.
- Reweight the gradient contributions by the **absolute change in the target metric** caused by swapping two documents:
  \[
  \lambda_{ij} \propto \frac{\partial \mathcal{C}}{\partial s_i}\cdot |\Delta NDCG_{ij}|
  \]
- This prioritizes corrections near the top ranks (position-sensitive).

**LambdaMART**:
- Uses **Gradient Boosted Decision Trees** (MART = Multiple Additive Regression Trees) instead of neural nets.
- Performs functional gradient boosting using LambdaRank-style gradients.
- Commonly a very strong baseline in IR benchmarks.

See also: [[gbdt]], [[decision_trees]], [[boosting]].

---

## Practical: Training LambdaMART with LightGBM (example workflow)

The source provides a worked example using **LightGBM**’s `lambdarank` objective and the **MSLR-WEB30K** dataset (a retired commercial Bing labeling dataset from ~2010):

- Dataset characteristics (as stated):
  - ~3.7M documents grouped into ~30k queries
  - 136-dimensional feature vectors
  - relevance labels from 0 (irrelevant) to 4 (perfectly relevant)

Key practical details:
- LightGBM needs **query group sizes** (`set_group`) to compute ranking losses properly.
- Example parameters:
  - `objective: "lambdarank"`
  - `metric: "ndcg"`
  - `ndcg_eval_at: [1,3,5,10]`
  - tree parameters like `num_leaves`, `learning_rate`, etc.

Feature importance observations from that example (dataset-era dependent):
- High split-importance features included:
  - **PageRank** and **site-level PageRank**
  - URL length / number of slashes (proxy for site quality at the time)
  - BM25-derived features (e.g., title BM25, document BM25)
- By gain-importance, a **query–URL click count** feature was highly influential.

> Interpretation note: these feature importances reflect the **2010-era feature set** and the specific dataset; modern production systems may rely more heavily on learned representations, though classical GBDT rerankers remain widely used.

---

## Unbiased / counterfactual LTR (learning from clicks)

Human labels are expensive, so many systems attempt to learn directly from clicks and other behavioral logs. The source emphasizes that click-based training is hard because clicks are **biased observations**.

### Common click biases
- **Position bias**: top-ranked results receive more examination and clicks (exposure depends on rank).
- **Selection bias**: some items have effectively zero chance to be examined (e.g., users rarely go to page 2+).
- **Trust bias**: users may click top results because they *trust* the ranking system, even if not truly relevant.

Operational takeaway from the source:
- **UI/UX changes** (SERP redesign) can change click behavior; bias estimators may need recalibration after major design updates.

### Counterfactual evaluation framing
Counterfactual LTR often uses the “policy” language:

- **Behavior policy** \(f_{\text{deploy}}\): the ranker that produced the logged data.
- **Evaluation policy** \(f_\theta\): the candidate ranker being evaluated/trained from the logs.

Goal:
- Estimate performance of \(f_\theta\) using data collected under \(f_{\text{deploy}}\), correcting for bias (see [[counterfactual_learning]]).

---

## Theoretical notes: does LambdaRank optimize a global loss?

The source highlights an open theoretical question historically discussed in the literature:

- While LambdaRank/LambdaMART work well empirically, it has been debated whether the gradient modifications correspond to optimizing a single global objective across iterations, because the sorting operation changes which pairs matter.

Reported/mentioned results:
- **Donmez et al. (2009)**: empirical evidence of local optimality (one-sided Monte Carlo test).
- **Burges et al. (2006)**: attempted arguments using differential geometry (Poincaré lemma); global objective existence across iterations remains subtle/unclear.

---

## LambdaLoss: a probabilistic framework connecting metrics and losses (Wang et al. 2019)

The source introduces **LambdaLoss**, a framework that provides a more explicit probabilistic interpretation:

- Treat the ranked list \(\pi\) as a latent variable.
- Define likelihood of labels given scores by marginalizing over permutations:
  \[
  P(\mathbf{y}\mid \mathbf{s}) = \sum_{\pi \in \Pi} P(\mathbf{y}\mid \mathbf{s}, \pi)\,P(\pi\mid \mathbf{s})
  \]
- Optimize negative log-likelihood:
  \[
  \mathcal{L}(\mathbf{y}, \mathbf{s}) = -\log P(\mathbf{y}\mid \mathbf{s})
  \]
- Can be optimized via an EM-like process conceptually (E-step over \(\pi\), M-step updating model parameters).

Key connection stated:
- Under particular choices of likelihood and a “hard assignment” approximation for \(P(\pi\mid \mathbf{s})\), **LambdaRank can be viewed as an EM procedure optimizing a LambdaLoss objective**.

This positions LambdaLoss as a unifying way to derive metric-aware ranking losses (particularly for NDCG-like gain/discount structure).

---

## Contradictions / tensions to be aware of (explicitly noted)

Because this is a new page, there is no internal contradiction with prior content. However, the **source itself** contains a few important tensions and caveats worth recording:

- **“Classical” vs “modern” retrieval**: The source describes embedding-based retrieval (often neural) as common at web scale, while also emphasizing classical lexical methods (TF‑IDF/BM25). This is not a contradiction, but a reminder that production retrieval is typically **hybrid**.
- **LambdaRank theoretical status**:
  - The source reports uncertainty about existence of a global loss across iterations for LambdaRank (historical concern),
  - while also stating that LambdaLoss (Wang et al. 2019) provides a framework in which LambdaRank optimizes an upper bound / can be derived under assumptions.
  - These are consistent but reflect a shift from “unclear objective” to “objective exists under a particular framework/approximation”.

---

## How this fits under “Classical ML Algorithms”

Within the broader taxonomy of classical ML, the LTR stack prominently features:

- **Logistic regression–like objectives** (pairwise cross-entropy; RankNet) — [[logistic_regression]]
- **Gradient boosting decision trees** (LambdaMART / LightGBM) — [[gbdt]]
- **Feature engineering** as a first-class component (URL features, BM25 signals, PageRank-derived signals)
- **Offline evaluation metrics** guiding modeling choices (NDCG, ERR, MAP, MRR)

In many real systems, these classical components coexist with neural retrieval/embedding stages (vector index), with classical ML often dominating the final reranking layer due to robustness, interpretability, and strong tabular performance.

---