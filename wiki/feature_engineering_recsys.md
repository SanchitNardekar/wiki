---
slug: feature_engineering_recsys
sources:
- hav4ik.github.io
tags: []
title: Feature Engineering for RecSys
updated: '2026-04-14'
---

# Feature Engineering for RecSys

Feature engineering in recommender systems (RecSys) is the process of extracting, transforming, and organizing signals about **users**, **items/documents**, and **context** into model-consumable features. In many production RecSys stacks—especially those that include a **retrieval → ranking → re-ranking** pipeline—feature engineering is a primary lever for improving final ranking quality: *the more expressive your features are, the better your ranking layer will perform*.

This page focuses on feature engineering for ranking-centric RecSys (often framed as **Learning to Rank (LTR)** in web search), while noting where the same ideas apply to broader recommendation scenarios. See also [[learning_to_rank]] and [[search_ranking]] if those pages exist; otherwise consider creating them.

---

## Where features live in a modern RecSys/search stack

A common abstraction (especially in web search, but also in large-scale recommendation) is:

- **Offline indexing / feature extraction**
  - Extract meaningful features/signals from all items (“documents”) continuously and store them.
- **Top-k retrieval (a.k.a. matching / “level-0 ranking”)**
  - Use lightweight signals to retrieve a candidate set of potentially relevant items.
- **Ranking / re-ranking (LTR)**
  - Use richer, more expensive features to produce the final ordering.

In this stack, features typically appear in multiple “indexes” or stores:

- **Inverted index**
  - Maps terms → documents for keyword/entity matching and term-based scoring (e.g., TF‑IDF, BM25).
- **Vector index**
  - Stores learned embeddings for documents and queries for embedding-based retrieval (kNN by cosine/euclidean similarity), often implemented with ANN methods for scale.
- **Feature index**
  - Stores thousands of engineered signals and (sometimes) compressed neural features used in re-ranking.

**Cross-reference:** retrieval-related embeddings and ANN indexing are closely related to [[embedding_retrieval]] and [[approximate_nearest_neighbors]].

---

## Types of features used in RecSys/LTR

Feature engineering often splits naturally by *what entity* the signal describes and *what stage* uses it.

### 1) Query–item (or user–item) matching features

These are the core “relevance” signals: how well does an item match the request/intent?

Examples from web search ranking features:

- Term-based relevance signals:
  - **BM25** features such as:
    - *Title BM25*
    - *Whole document BM25*
- Query–document pair interaction signals:
  - **Query–URL click count** (historical interaction strength for that pair)

In RecSys, analogous features include:

- user–item co-occurrence statistics
- query-text ↔ item-text similarity
- embedding similarity features (dot product, cosine, calibrated distances)
- semantic match model scores (e.g., cross-encoder / dual-encoder outputs)

**Note:** term-based features (BM25/TF‑IDF) are classically “retrieval” features, but in practice are often carried forward into ranking as explicit features.

---

### 2) Item/document quality & authority features

These are query-independent (or weakly query-dependent) signals describing item quality.

Examples observed in classic web search feature sets:

- **PageRank**
- **Site-level PageRank**
- Learned quality classifier outputs:
  - *QualityScore*, *QualityScore2* (webpage quality model outputs)

In RecSys, analogs include:

- item popularity / sales / ratings (with care about feedback loops)
- content quality classifiers
- creator quality signals
- freshness/recency and decay features

---

### 3) URL/content structure features (domain-specific)

Some surprisingly effective historical features in search datasets include:

- **Length of URL**
- **Number of slashes in URL**

These likely correlate with quality (e.g., shorter, cleaner URLs historically associated with higher-quality sites). In non-web RecSys, similar “structure” proxies might include:

- metadata completeness
- number of images/attributes
- description length, title length
- categorical hygiene (missingness patterns)

**Caution:** structure features can be brittle and can encode spurious correlations that drift over time.

---

### 4) Behavioral and interaction features

Behavioral features are derived from user interactions:

- Click-through signals (CTR-like)
- Query–item click counts
- Conversion signals (purchases, sign-ups, profit)
- Engagement measures (dwell time, completion, likes)

In the LTR framing, behavioral data is often used for:

- **Counterfactual LTR** (learning from logged interactions)
- **Online LTR** (learning from live interaction streams)

**Important:** user interaction signals are **biased** (see “Biases in behavioral features” below), so using them directly as labels or features can introduce systematic errors.

---

## Feature engineering for Learning to Rank (LTR)

LTR treats ranking as learning a function that scores items for a query (or user context), then sorts by score.

A standard representation is:

- For each query–document pair \((q, d)\), compute a feature vector \(\mathbf{x}_{d}\)
- A model \(f_\theta(\cdot)\) outputs a score \(s_i = f_\theta(\mathbf{x}_i)\)
- Items are sorted by \(s_i\)

Common LTR models consume:

- dense and sparse relevance signals (BM25/TF‑IDF, embedding similarities)
- authority/quality signals (e.g., PageRank)
- behavioral aggregates (click counts, conversions)
- context signals (device, locale, time, user cohort)—more common in RecSys than pure web search

**Cross-reference:** metrics and objectives used by LTR are closely tied to [[ranking_metrics]] and [[ndcg]].

---

## Metrics influence what features matter

Ranking metrics used in information retrieval (and many RecSys ranking evaluations) are position-sensitive and reward getting the *top of the list* correct. Common offline metrics include:

- **NDCG@T** (Normalized Discounted Cumulative Gain): the most common
- **ERR** (Expected Reciprocal Rank)
- **MRR**, **MAP** (often used for retrieval; less suited when graded relevance is essential, unless adapted)

### NDCG definition (for graded relevance)

For truncation \(T\) and label \(l_i\) at rank \(i\):

- \(DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log(1+i)}\)
- \(NDCG@T = \frac{DCG@T}{\max DCG@T}\)

**Practical implication for features:** if you optimize or evaluate with NDCG/ERR, feature engineering should prioritize signals that disambiguate the *top few results* (e.g., better semantic match, better quality signals, better personalization/context).

---

## Behavioral features are biased (and design-dependent)

If you use click/conversion data as labels or features, biases matter.

Common biases in click signals:

- **Position bias**
  - Users examine and click top-ranked results more, independent of true relevance.
- **Selection bias**
  - Some items have near-zero probability of being examined (e.g., results on page 2+).
- **Trust bias**
  - Users may trust the system and click higher-ranked items even if not relevant.

**Design dependency:** UI changes (layout, snippet style, pagination, ads) can substantially change examination/click patterns. Position-bias estimators should be re-validated after major design changes.

**Cross-reference:** this connects to [[counterfactual_learning_to_rank]] and [[unbiased_learning_to_rank]].

---

## Examples of engineered features from a classic LTR dataset (MSLR-WEB30K)

The MSLR-WEB30K dataset (released 2010; based on Bing labeling data) contains:

- ~3.7M documents grouped into ~30k queries
- 136-dimensional feature vectors
- relevance labels from 0 (irrelevant) to 4 (perfectly relevant)

A LambdaMART model trained with LightGBM can be inspected via feature importance, illustrating what feature engineering looked like pre-deep-learning in major search engines.

Notable “important” features reported:

- **Authority/quality**
  - PageRank (#130)
  - Site-level PageRank (#131)
  - QualityScore / QualityScore2 (#132, #133)
- **Document/query matching**
  - Title BM25 (#108)
  - Whole document BM25 (#110)
- **Behavioral**
  - Query–URL click count (#134) (high by gain importance)
- **Structure**
  - URL length (#127)
  - Number of slashes in URL (#126)

**Takeaway:** strong ranking performance often comes from combining:
- relevance matching (BM25 / semantic features),
- global quality (PageRank / quality classifiers),
- behavioral aggregates (click counts),
- and domain-specific proxies (URL structure).

---

## Retrieval-stage feature engineering (candidates matter)

Even the best ranking features cannot help if retrieval fails to include relevant candidates. In web-scale stacks:

- retrieval often uses a **hybrid** of:
  - keyword/entity matching (inverted index) and
  - embedding-based retrieval (vector index)
- approximate nearest neighbors (ANN) is typically used for scalability (aiming for near \(O(1)\) retrieval behavior in practice vs. slow exact methods at scale)

Feature engineering for retrieval includes:

- building robust query/item embeddings (text, image, multimodal)
- query expansion signals (e.g., knowledge graph expansions in some systems)
- calibration features for blending lexical and semantic retrieval scores

**Cross-reference:** see [[candidate_generation]] and [[two_tower_models]] for embedding retrieval patterns (if available).

---

## Model choice affects feature engineering (and vice versa)

### Tree-based rankers (e.g., LambdaMART / LightGBM)

- Benefit greatly from:
  - well-crafted numerical/categorical aggregates
  - monotonic/threshold-friendly signals (counts, BM25, quality scores)
  - missingness indicators
- Feature importance tools (split/gain) help iterate on feature sets.

### Neural rankers

- Often shift effort toward:
  - representation learning (embeddings, cross-encoders)
  - learned interaction features
- Still benefit from explicit engineered signals (quality, popularity, freshness, policy/business constraints).

---

## Contradictions & tensions to be aware of

Because this is a new page (no prior content), there are no direct contradictions with earlier page statements. However, the source material implies some **important tensions** that commonly surface in RecSys feature engineering:

- **“Clicks measure relevance” vs. “clicks are biased”**
  - Clicks/CTR are cheap relevance proxies, but position/trust/selection bias can dominate observed click patterns.
  - Resolution: unbiased LTR / counterfactual methods, careful logging, randomized interventions, bias modeling.
- **“More features is better” vs. “features can encode spurious correlations”**
  - Features like URL length can correlate with quality historically, but may fail under distribution shift or be gamed.
  - Resolution: robustness checks, monitoring, ablations, and drift-aware validation.

(If future sources contradict specific best practices—e.g., whether PageRank still matters materially—add explicit “contradiction notes” here with citations.)

---

## Practical checklist (feature engineering in ranking-focused RecSys)

- **Define the stage**
  - retrieval vs ranking vs re-ranking: different latency budgets and feature availability.
- **Organize features by entity**
  - user/query, item/document, user–item/query–item interactions, context.
- **Include multiple signal families**
  - lexical relevance, semantic relevance, quality/authority, behavioral aggregates, freshness.
- **Guard against bias**
  - treat clicks/conversions as biased; model exposure and UI effects where needed.
- **Optimize for the metric**
  - NDCG/ERR emphasize top ranks; craft features that disambiguate the head.
- **Monitor drift**
  - UI changes, content shifts, and adversarial behavior can invalidate brittle proxies.

---

## Related pages

- [[learning_to_rank]]
- [[unbiased_learning_to_rank]]
- [[counterfactual_learning_to_rank]]
- [[ranking_metrics]]
- [[ndcg]]
- [[candidate_generation]]
- [[embedding_retrieval]]
- [[approximate_nearest_neighbors]]