---
slug: search_and_retrieval
sources:
- hav4ik.github.io
tags: []
title: Search and Information Retrieval
updated: '2026-04-14'
---

# Search and Information Retrieval

Search and Information Retrieval (IR) studies how to **retrieve** and **rank** items (commonly called *documents*) that satisfy a user’s *query*. In modern systems, “documents” can be web pages, products, videos, books, images, or any other content type.

This page focuses on how search engines work at a high level and on **Learning to Rank (LTR)**—machine-learned ranking models that order results by relevance.

Related pages to link with as the wiki grows:
- [[learning_to_rank]]
- [[information_retrieval]]
- [[search_engine_architecture]]
- [[ranking_metrics]]
- [[bm25]]
- [[tf_idf]]
- [[knowledge_graphs]]
- [[approximate_nearest_neighbors]]
- [[bert]]
- [[lightgbm]]
- [[counterfactual_learning]]
- [[online_learning_to_rank]]
- [[unbiased_learning_to_rank]]
- [[click_models]]

---

## Core concepts

- **Query**: the user’s input (text, image, etc.).
- **Document**: an item that can be retrieved and shown.
- **Relevance**: how well a document satisfies the user’s information need.
- **Retrieval vs. ranking**:
  - **Retrieval (matching / candidate generation)** finds a *set* of potentially relevant documents efficiently.
  - **Ranking (re-ranking)** orders those candidates to maximize user-perceived relevance (and sometimes business objectives).

---

## How modern search engines work (abstract pipeline)

At a very high level, modern search engines share a common skeleton:

1. **Indexing (offline, continuous)**
   - Crawl/ingest documents and compute useful signals/features.
   - Store them in one or more indexes/datastores.

2. **Top-*k* retrieval (“Level-0 ranking” / matching)**
   - Given a query, retrieve a manageable candidate set (*k*) quickly.

3. **Ranking / re-ranking**
   - Use richer features and more expensive models to sort candidates by relevance (and optionally personalization signals).

This “simple” schema hides massive real-world complexity; web-scale systems are substantially more complicated and typically involve multiple stages, multiple models, and significant infrastructure.

See also: [[search_engine_architecture]]

---

## Indexing and index types

Indexing extracts and stores signals that later stages can use.

Common index types mentioned in web search and large-scale retrieval:

- **Inverted index** (posting lists)
  - Maps terms → documents containing the term.
  - Supports classic lexical scoring such as [[tf_idf]] and [[bm25]].
- **Vector index**
  - Stores dense embeddings for documents (and sometimes queries).
  - Enables *embedding-based retrieval* via nearest-neighbor search in embedding space.
  - Embeddings may come from BERT-like models for text (see [[bert]]) or contrastive vision-language models for visual search.
- **Feature index**
  - Stores a large set of engineered signals (potentially thousands), including compressed neural features.
  - Used heavily in later re-ranking stages; feature engineering strongly influences ranker quality.

---

## Retrieval (candidate generation)

Retrieval aims for speed and recall: include most of the relevant documents in the candidate set.

Techniques:

- **Keyword/entity matching** using the inverted index
- **Embedding-based retrieval** using a vector index
  - Compute an embedding for the query.
  - Retrieve the *k* nearest document embeddings (cosine similarity or Euclidean distance).
- **Hybrid retrieval**
  - Web-scale engines often combine lexical and embedding retrieval.
- **Query expansion**
  - General-purpose engines may use [[knowledge_graphs]] to expand queries and retrieve additional relevant results.

### Approximate Nearest Neighbors (ANN) at scale
Exact metric-tree methods (e.g., k-d trees) are typically not used at web-scale due to runtime and memory constraints; instead, large systems use **Approximate Nearest Neighbor** methods to achieve near-constant-time behavior in practice.

See: [[approximate_nearest_neighbors]]

---

## Ranking and Learning to Rank (LTR)

Ranking is the stage that “makes search work”: candidates are sorted by predicted relevance to the query (and optionally user context/preferences).

- Smaller engines may rely on heuristics/rules.
- Major search engines generally use **machine-learning-based ranking**, i.e., **Learning to Rank (LTR)**.

Historical note:
- PageRank was once a dominant signal for early Google ranking, but over time ranking evolved to incorporate many more signals and models.
- As of 2020, PageRank is described as still present but only a small part of Google’s broader system.

See: [[learning_to_rank]]

---

## Learning to Rank problem setup

Given:
- a query \( \mathbf{q} \)
- a set of retrieved documents \( \mathcal{D} = \{d_1, \ldots, d_n\} \)

Learn a function \( f(\mathbf{q}, \mathcal{D}) \) (often implemented as scoring each document independently) that produces an ordering of documents, ideally with the most relevant first.

Typical approach:
- Compute feature vectors \( \mathbf{x}_{(\mathbf{q}, d)} \) for each query-document pair.
- Predict scores \( s_i = f_\theta(\mathbf{x}_i) \).
- Sort documents by score.

---

## What does “relevance” mean in practice?

Relevance can be estimated with multiple signals; common components include:

- **Human-labeled relevance**
  - Human judges assign graded relevance scores (e.g., 1–5) using guidelines.
  - This is expensive but high quality.
- **Click-through rate (CTR)**
  - Cheap implicit feedback: how often users click a result.
  - Known to be biased by presentation and ranking (see *Click biases* below).
- **Conversion rate / business outcomes**
  - Especially in e-commerce: purchases, sales, profit, etc.
  - Example: conversion rate = buys / searches.

**Potential tension / contradiction to be aware of (needs future sourcing):**
- Human-judged “relevance” can diverge from click-based or conversion-based objectives (e.g., users click sensational items; conversions may favor promoted or high-margin items). The source text frames relevance as a combination of these factors, but in many systems these are treated as *distinct* objectives or labels. (No direct contradiction within the provided text; this is a modeling caveat to note explicitly.)

---

## Flavors of Learning to Rank methods

LTR methods are commonly categorized as:

### Offline vs. online
- **Offline LTR**
  - Train on a fixed dataset offline.
- **Online LTR**
  - Learn from user interactions in real time; model updates based on live feedback.

See: [[online_learning_to_rank]]

### Supervised vs. counterfactual
- **Supervised LTR**
  - Uses human-labeled relevance judgments.
- **Counterfactual LTR**
  - Learns from historical logged interactions (clicks, engagement, conversions).
  - Must account for bias in observed feedback.

See: [[counterfactual_learning]] and [[unbiased_learning_to_rank]]

### Pointwise vs. pairwise vs. listwise (supervised objectives)
- **Pointwise**: loss considers one document at a time (classification/regression).
- **Pairwise**: loss considers document pairs (preference learning).
- **Listwise**: loss considers the entire ranked list directly.

---

## Ranking metrics (offline evaluation)

Common IR ranking metrics include:

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **ERR** (Expected Reciprocal Rank)
- **NDCG** (Normalized Discounted Cumulative Gain)

A noted distinction from the source:
- MAP and MRR are widely used for retrieval, but are often considered less suitable for graded relevance ranking unless adapted, because they do not inherently incorporate graded relevance labels.

See: [[ranking_metrics]]

### DCG and NDCG

For truncation level \(T\), relevance labels \(l_i\):

\[
DCG@T = \sum_{i=1}^T \frac{2^{l_i}-1}{\log(1+i)}
\]

\[
NDCG@T = \frac{DCG@T}{\max DCG@T}
\]

NDCG is widely used because it:
- supports graded relevance
- emphasizes high ranks via the logarithmic discount

### ERR (Expected Reciprocal Rank)

ERR models a user scanning down the ranked list until satisfied:

\[
ERR = \sum_{r=1}^n \frac{1}{r} R_r \prod_{i=1}^{r-1}(1-R_i), \quad
R_i = \frac{2^{l_i}-1}{2^{l_m}}
\]

---

## Key supervised LTR methods

The source emphasizes a historically influential line of work led by Christopher Burges (Microsoft Research):

- **RankNet** (Burges et al., 2005)
  - Pairwise approach; formulates ranking as optimizing a differentiable objective with gradient descent.
- **LambdaRank** and **LambdaMART** (Burges et al., 2006)
  - Modify gradients to more directly optimize position-sensitive metrics like NDCG.
  - **LambdaMART** uses gradient-boosted decision trees (MART = Multiple Additive Regression Trees) instead of neural networks.
  - As of the time of writing in the source, LambdaMART remains a strong baseline and can outperform newer methods on some benchmarks.

### RankNet (pairwise cross-entropy)

Model probability that \(d_i\) should rank above \(d_j\):

\[
P_{ij} = \frac{1}{1+e^{-\sigma(s_i-s_j)}}
\]

Cost (cross entropy) with target \(\widetilde{P}_{ij}\):

\[
\mathcal{L}_{\text{RankNet}}(s_i,s_j)
= -\widetilde{P}_{ij}\log P_{ij} - (1-\widetilde{P}_{ij})\log(1-P_{ij})
\]

### ListNet (listwise via Plackett–Luce)

ListNet (Cao et al., 2007) defines a probability distribution over permutations (ranked lists) and optimizes cross-entropy between distributions induced by labels and model scores.

(Implementation detail: ListNet is often approximated in practice due to the combinatorics of full permutations; the source presents the core Plackett–Luce-based formulation.)

### LambdaRank and LambdaMART (metric-shaped gradients)

Problem:
- Metrics like NDCG/ERR require sorting and are not differentiable with respect to scores.

LambdaRank idea:
- Start from pairwise gradients and scale them by the *change in the target metric* if two documents swap positions.

For NDCG:

\[
\lambda_{ij} \equiv \frac{\partial \mathcal{C}}{\partial s_i}\cdot |\Delta NDCG_{ij}|
\]

with

\[
\Delta NDCG_{ij} =
\frac{2^{l_j}-2^{l_i}}{\max DCG@T}
\left(\frac{1}{\log(1+i)}-\frac{1}{\log(1+j)}\right)
\]

LambdaMART:
- applies the same idea but uses boosted trees and performs gradient boosting in function space.

See also: [[lightgbm]] (commonly used LambdaMART implementation)

---

## Practical example: training LambdaMART with LightGBM

The source outlines a practical training flow using **LightGBM’s** `lambdarank` objective and the **MSLR-WEB30K** dataset (a retired Microsoft Bing learning-to-rank dataset, published 2010):

- ~3.7M documents
- ~30K queries
- 136 features per document
- relevance labels 0–4

Key operational detail:
- LightGBM needs **query group sizes** (how many documents per query) to compute ranking losses/metrics.

### Observed feature importance (from the source’s example)
On MSLR-WEB30K (2010-era features), important signals include:

- PageRank-related features (e.g., site-level PageRank and PageRank)
- URL structure features (URL length, number of slashes)
- Web page “quality score” features from a classifier
- Lexical retrieval features like BM25 over title and full document (see [[bm25]])
- Click-count feature (query–URL click count) shows strong importance by gain

**Interpretation note:**
- This reflects the dataset’s time period and feature set; modern production systems also incorporate deep neural features and embeddings much more heavily (consistent with the earlier “vector index” and “feature index” discussion).

---

## Theoretical notes: does LambdaRank optimize a “real” loss?

The source highlights open questions and partial answers:

- **Convergence / existence of a global loss**
  - Donmez et al. (2009) empirically test local optimality using a one-sided Monte Carlo test and find evidence of convergence to a local minimum.
- **Poincaré lemma argument (Burges et al., 2006)**
  - Suggests conditions under which a global potential function might exist, but due to sorting and recomputation of lambdas across iterations, a global loss across iterations is not straightforwardly established.
- **LambdaLoss framework (Wang et al., 2019)**
  - Provides a probabilistic framework where LambdaRank is interpreted as an EM-like procedure optimizing a well-defined negative log-likelihood (or an upper bound thereof), connecting “lambda” gradients to an underlying objective.

See: [[learning_to_rank]]

---

## Unbiased Learning to Rank (learning from user behavior)

Human labeling is expensive; a natural alternative is to learn from user interactions (especially clicks). However, click signals are **biased**.

This motivates:
- **Counterfactual Learning to Rank**
- **Unbiased Learning to Rank**

See: [[unbiased_learning_to_rank]] and [[counterfactual_learning]]

### Click signal biases

Common biases in implicit feedback:

- **Position bias**
  - Users examine top-ranked results more; higher ranks receive disproportionate attention and clicks.
  - Eye-tracking studies illustrate this, and **UI/design changes can change the bias profile**, requiring recalibration of bias estimators.
- **Selection bias**
  - Some results have near-zero probability of being examined (e.g., results beyond the first page).
  - Important distinction: some methods can correct for position bias only if selection bias is not severe.
- **Trust bias**
  - Users may trust the system’s ordering and click top results even when not most relevant.
  - Related to but distinct from position bias (trust is about *belief in ranking quality*, not just examination likelihood).

See: [[click_models]]

### Counterfactual evaluation (core idea)

Goal:
- Evaluate a new ranker \(f_\theta\) using logs collected under a previously deployed ranker \(f_{\text{deploy}}\) (often called a **behavior policy**).

Terminology:
- **Behavior policy**: the deployed system that generated the logged rankings.
- **Evaluation policy**: the new candidate system being evaluated using those logs.

**Important caveat (inherent limitation):**
- Because logs are biased by the behavior policy and by presentation biases, counterfactual methods require explicit bias modeling/correction to avoid learning or evaluating the wrong thing.

---

## Notes on scope and limitations

- The “index → retrieve top-*k* → rank” pipeline is intentionally simplified; production search stacks include multiple retrieval/ranking stages, blending, deduplication, safeguards, policy constraints, and more (see [[search_engine_architecture]]).
- Many ideas here generalize beyond web search to internal search (e.g., retail, media, social), where “documents” are products, videos, posts, etc.

---

## Contradictions and open questions (tracked)

Since this is a new page and only one source has been integrated so far, there are **no direct contradictions with existing page content**.

Open questions / areas likely to require reconciliation as more sources are added:
- Whether “relevance” should be treated as a single concept combining human labels, CTR, and conversions (as the source describes) versus a multi-objective setup where these are distinct targets.
- The exact role and prevalence of specific algorithms in current major search engines (claims like “PageRank is still a small part” are time- and source-dependent and may vary by engine and by interpretation).