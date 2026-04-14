---
slug: search_and_retrieval
sources:
- hav4ik.github.io
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
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
- (New, suggested) [[metric_learning]]
- (New, suggested) [[contrastive_learning]]
- (New, suggested) [[vector_search]]

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

### Two-tower embedding indexes (industrial pattern)
A common industrial pattern (especially in large-scale retrieval and *pre-ranking*) is to learn embeddings with a **two-tower / dual-encoder** model, then store (typically) the *document/item* tower embeddings in a vector index for fast similarity search:

- Two separate encoders produce:
  - a **query/user embedding** \( \mathbf{e}_q \)
  - a **document/item embedding** \( \mathbf{e}_d \)
- A similarity score is computed with a cheap function, often **inner product**:
  - \( s(q,d) = \langle \mathbf{e}_q, \mathbf{e}_d \rangle \)

Operational notes from the existing page content:
- The two towers can be computed **independently in parallel** and only interact at the output (“**late interaction**”).
- The *item/document tower embeddings* are often **frozen after training** and served from an **indexing service** for efficient inference.
- Some systems may even freeze both towers and retrain offline periodically (e.g., daily) while rebuilding indexes.

Additional note from the new source (Cosine Adapter paper, CIKM’24):
- Many dense retrieval systems use **cosine similarity** \( \cos(\mathbf{e}_q,\mathbf{e}_d) \) for ANN search and top-*K* candidate retrieval, which can create downstream challenges for *filtering* (see “Relevance filtering” below).

(These ideas connect directly to the “Vector index” section and to [[approximate_nearest_neighbors]] for scalable nearest-neighbor retrieval, and practically to [[vector_search]].)

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

### Dense retrieval has “no natural cutoff” (new source)
The new source highlights an important operational difference between lexical and dense retrieval:

- **Lexical retrieval** (e.g., BM25 on an inverted index) *implicitly* limits candidates because documents must match query terms.
- **Dense retrieval via ANN** often has **no natural cutoff**: ANN will return top-*K* even if many are irrelevant, especially when the query has very few truly relevant items (common in **e-commerce product search**).

**Practical consequence:** maximizing recall at retrieval time can yield low precision candidate sets, increasing load on later re-rankers and potentially harming UX if irrelevant items appear prominently.

This motivates adding an explicit **relevance filtering** step between ANN retrieval and downstream ranking (see below).

---

## Multi-stage / cascade ranking systems (retrieval → pre-ranking → ranking)

The existing “retrieve top-*k* → rank” pipeline is a simplification. The existing page already emphasizes a widely used **multi-stage (cascade) ranking** architecture:

- Real systems can have **tens of millions** of candidate items/documents.
- Strict **latency constraints** strongly shape architecture; the source claims:
  - even ~**100ms** additional response time can measurably degrade user experience and revenue (time- and product-dependent).
- Because a single complex model cannot score every candidate within latency limits, systems typically adopt stages such as:
  - **Recall / retrieval**: very fast, recall-oriented
  - **Pre-ranking**: filters the retrieved set further using fast learned models (often deep but efficient)
  - **Ranking / re-ranking**: the most expensive models/features applied to the smallest candidate set

In this framing:
- Earlier stages emphasize **efficiency and recall metrics**.
- Later stages emphasize **effectiveness** using richer interaction modeling.

See also: [[search_engine_architecture]]

### Relevance filtering as a stage between ANN retrieval and re-ranking (new source)
The new source (“Relevance Filtering for Embedding-based Retrieval”, CIKM’24) adds a specific, production-motivated stage that often sits between **ANN retrieval** and **re-ranking**:

- **Relevance filtering (post-retrieval)**:
  - Operates on the **top-*K*** ANN results as a *post-processing step*.
  - Goal: remove obviously irrelevant candidates **before** expensive re-ranking.
  - Particularly helpful when:
    - the number of truly relevant items is small (e.g., product search),
    - ANN returns many semantically “near” but commercially irrelevant results,
    - re-rankers would otherwise spend compute demoting irrelevant items.

**Relationship to “pre-ranking”:**
- This filtering stage can be viewed as a lightweight **pre-ranking/relevance control** module (but it is conceptually distinct from “learned scoring over items” because it may transform/calibrate the *retrieval score* itself, rather than rescore with full query–item features).

---

## Ranking and Learning to Rank (LTR)

Ranking is the stage that “makes search work”: candidates are sorted by predicted relevance to the query (and optionally user context/preferences).

- Smaller engines may rely on heuristics/rules.
- Major search engines generally use **machine-learning-based ranking**, i.e., **Learning to Rank (LTR)**.

Historical note:
- PageRank was once a dominant signal for early Google ranking, but over time ranking evolved to incorporate many more signals and models.
- As of 2020, PageRank is described as still present but only a small part of Google’s broader system.

See: [[learning_to_rank]]

### Representation-based vs interaction-based neural rankers (where two-tower fits)
The existing page distinguishes families of neural ranking/matching models that often map onto different pipeline stages:

- **Two-tower / dual-encoder (bi-encoder)**: *representation-based*
  - Encodes query and document independently; matches via dot product or similar.
  - Favored for **retrieval** and **pre-ranking** due to speed and indexability.
- **Interaction-focused models**: compute richer query–document interactions
  - Examples mentioned in the existing page include:
    - DRMM, KNRM (interaction matrix + neural network)
    - **Cross-encoders** like [[bert]] that jointly encode query+document and model full token interactions
    - **ColBERT**-style “late interaction” that keeps query/document encodable but uses finer-grained interaction at scoring time

**Connection:** the new source strengthens a key operational point: dual encoders’ similarity scores (often cosine) are optimized for *relative* comparisons and may be poorly calibrated in an *absolute* sense, which affects not just ranking but also **filtering/truncation** decisions.

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

### Additional nuance: absolute vs relative “relevance scores” in dense retrieval (new source)
The new source emphasizes that for embedding-based retrieval, the **raw cosine similarity** is often *not* an interpretable “relevance score”:

- Dual encoders are commonly trained with:
  - **contrastive losses** (in-batch negatives), or
  - **softmax listwise ranking losses**
  (see also: [[contrastive_learning]])
- Both losses shape the space to enforce **relative ordering** (positive closer than negatives), but do **not** guarantee:
  - that a cosine value corresponds to an absolute probability of relevance, or
  - that cosine scores are **comparable across queries**.

This matters for any system design that tries to set global cutoffs (“only show results above score *t*”).

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

A noted distinction from the existing page content:
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

### Metrics for relevance filtering / truncation (new source)
The new source focuses on a different evaluation problem: **filtering/truncating** the retrieved set to improve precision with minimal recall loss.

Metrics used there include:
- **PR AUC** (Area Under the Precision–Recall Curve)
  - Evaluated *without* committing to a single threshold (measures separability of relevant vs irrelevant).
- **P@R95** (Precision at 95% Recall)
  - Choose a global threshold achieving 95% recall (relative to no filtering) and report precision.
- **Filter%**
  - Fraction of retrieved candidates removed.
- **Null%**
  - Fraction of queries with **zero results after filtering** (important UX guardrail).
- **MRR**
  - Used on MS MARCO as a standard retrieval metric.

**Connection:** this is adjacent to classic IR “ranked list truncation” work and complements [[ranking_metrics]] with filtering-oriented evaluation.

---

## Key supervised LTR methods

The existing page emphasizes a historically influential line of work led by Christopher Burges (Microsoft Research):

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

The existing page outlines a practical training flow using **LightGBM’s** `lambdarank` objective and the **MSLR-WEB30K** dataset (a retired Microsoft Bing learning-to-rank dataset, published 2010):

- ~3.7M documents
- ~