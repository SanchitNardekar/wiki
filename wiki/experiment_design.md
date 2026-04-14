---
slug: experiment_design
sources:
- relevance_filtering_for_embedding_based_retrieval.pdf
tags: []
title: Experiment Design and A/B Testing
updated: '2026-04-14'
---

# Experiment Design and A/B Testing

## Overview

Experiment design and A/B testing are structured methods for evaluating whether a change to a system causes measurable improvement on selected metrics. In online systems (e.g., search, recommendations, ads, and e-commerce), A/B tests are used to validate that offline improvements translate into real user impact.

This page focuses on practical experiment design for information retrieval (IR) and ranking systems, with particular attention to cases where **offline retrieval metrics (precision/recall)** interact with **online business/engagement metrics**.

See also: [[retrieval_models_and_ranking]], [[embedding_based_retrieval]], [[ranking]], [[offline_evaluation]], [[online_metrics]]

---

## Why A/B testing is necessary (especially in retrieval)

Offline evaluations often optimize for relevance (e.g., precision, recall, MRR), while online outcomes include user behavior and business KPIs (e.g., orders, revenue). Improvements in offline relevance can:

- Produce **neutral** online outcomes (no detectable KPI change)
- Improve user satisfaction without changing purchases (shorter time-to-find, fewer bad results)
- Improve or harm downstream stages (e.g., rerankers) depending on candidate set changes

A key example from dense retrieval in product search is relevance filtering: improving **precision of retrieved candidates** can reduce downstream compute and reduce irrelevant items, but it may also introduce a **recall loss**. Therefore, experiment design must explicitly manage and measure the trade-off.

---

## Running example: relevance filtering for embedding-based retrieval

A 2024 CIKM paper (“Relevance Filtering for Embedding-based Retrieval”) introduces a lightweight relevance filtering component (“Cosine Adapter”) for dense retrieval systems that use Approximate Nearest Neighbor (ANN) search. The paper demonstrates:

- **Offline gains** in precision-oriented metrics with small recall loss.
- **Online A/B testing** on Walmart’s site showing **improved judged precision** with **neutral engagement impact** (orders and GMV).

This example illustrates common A/B testing patterns in search:
- Offline improvements are validated with offline benchmarks.
- Online tests confirm real-world value and detect unintended side effects.

Related pages: [[approximate_nearest_neighbor_search]], [[dense_retrieval]], [[hybrid_retrieval]], [[reranking]], [[product_search]]

---

## Key experimental questions to answer

When designing an A/B test, define the causal questions precisely:

- **Does the change improve relevance?**
  - Often evaluated via human judgments or proxy metrics (CTR, reformulations).
- **Does it change engagement/business outcomes?**
  - Orders, revenue/GMV, conversions, add-to-cart, dwell time.
- **Does it change system behavior in harmful ways?**
  - Null-result rate, latency, timeouts, cost, diversity, long-tail coverage.
- **Where in the stack is the change applied (retrieval vs reranking)?**
  - Retrieval-stage changes can alter the candidate distribution and affect rerankers.

---

## Offline experiment design (pre-A/B)

### Common offline metrics (retrieval & filtering)

The relevance filtering work highlights a useful set of offline metrics for retrieval-stage changes:

- **PR AUC (Area Under Precision–Recall Curve)**
  - Computed without deploying a hard filter; evaluates score separability across thresholds.
- **P@R95 (Precision at 95% Recall)**
  - Choose a *global threshold* to maintain 95% recall relative to no filtering; report precision.
  - Useful for changes that should “mostly preserve recall” while increasing precision.
- **Filter%**
  - Fraction of retrieved items filtered out; helps quantify candidate set reduction and compute savings.
- **Null%**
  - Fraction of queries that return zero results after filtering; critical guardrail metric in search.
- **MRR (Mean Reciprocal Rank)**
  - Standard retrieval metric (used in MS MARCO experiments in the paper).

Related pages: [[precision_and_recall]], [[auc]], [[mrr]], [[thresholding]]

### Baselines and comparisons

For filtering dense retrieval outputs, the paper compares:

- Global threshold on **raw cosine similarity**
- Global threshold on **max-normalized cosine similarity** (per-query normalization by max score)
- A ranked-list truncation model (**Choppy**) that predicts a cutoff position per query

This suggests a general offline design principle:
- Compare against *simple global thresholds* and *state-of-the-art learned truncation* methods when applicable.

---

## Online A/B testing design for search/retrieval changes

### Unit of randomization

Common choices:
- **User-level randomization** (reduces interference across sessions)
- **Session-level randomization** (useful when user identity is unstable)
- **Query-level randomization** (can be problematic due to repeated exposure and user learning)

For retrieval and ranking, user-level is typically preferred when feasible.

### Primary metrics vs guardrails

**Primary metrics** (business/engagement):
- Orders
- GMV / revenue
- Conversion rate

**Relevance metrics** (often secondary or diagnostic):
- Human-judged precision in top-K results (post-reranker)
- CTR, add-to-cart, long-click, abandonment

**Guardrails**:
- Null-result rate (also see offline Null%)
- Latency and error rates
- Excessive filtering or candidate starvation for rerankers

---

## Case study: Walmart production A/B test (Cosine Adapter relevance filtering)

### System context

- Walmart uses **hybrid retrieval** combining lexical retrieval and embedding-based retrieval.
- The relevance filtering module is inserted between ANN retrieval and reranking, filtering low-relevance candidates using a calibrated score and a global threshold.
- In production, the deployed mapping used the **square root transformation** via the Cosine Adapter.

Cross references: [[hybrid_retrieval]], [[lexical_retrieval]], [[dense_retrieval]], [[reranking]]

### Experiment setup (as reported)

- Evaluated impact on:
  - **Relevance** (human judgment on top-10 products shown to users, post-reranker)
  - **Engagement** (online A/B test)
- Threshold tuned to **99% recall** (more conservative than 95% used in offline metric definitions) to minimize risk of filtering relevant products.

**Note on design choice:** tuning to 99% recall reflects a common production A/B practice: prioritize safety/guardrails when recall loss is costly (e.g., causing null results or missing key products).

### Results (as reported)

**Human-judged relevance (impacted queries):**
- Precision lift:
  - Top-5: **+5.34%** (p-value 0.00)
  - Top-10: **+4.00%** (p-value 0.00)

**Online business/engagement:**
- Orders lift: **+0.03%** (p=0.86) → statistically neutral
- GMV lift: **-0.11%** (p=0.83) → statistically neutral

#### Interpretation for experiment design

- The experiment demonstrates a frequent outcome in search systems:
  - **Relevance improves measurably**, yet **business KPIs remain neutral** within measurement noise.
- This supports designing A/B tests with:
  - Both business metrics and relevance diagnostics
  - Adequate power/duration to detect small KPI changes if expected
  - Clear decision policy (e.g., allow shipping if relevance improves and KPIs are neutral, provided guardrails are satisfied)

---

## Contradictions and tensions to track explicitly

Because this page currently integrates a retrieval-focused source, there are important *general experiment design tensions* highlighted by the paper:

- **Goal tension: maximize recall vs improve precision**
  - Dense retrieval often optimizes recall, but user experience can degrade with low precision.
  - Filtering improves precision but risks recall loss.
- **Score interpretability**
  - The paper argues raw cosine similarity is **not comparable across queries**, making a single global threshold on cosine similarity unreliable.
  - This conflicts with a naïve experimental assumption that a score threshold works uniformly across all traffic.
- **Offline vs online outcomes**
  - Offline precision gains do not guarantee orders/GMV gains; online may remain neutral.

*(No direct contradiction with existing page content, since this is a new page.)*

---

## Practical checklist: designing an A/B test for retrieval-stage filtering

1. **Define success criteria**
   - Primary: orders/GMV/conversion (or your domain KPIs)
   - Secondary: judged precision@K, CTR@K, reformulation rate
2. **Add guardrails**
   - Null-result rate, latency, error rate, downstream reranker stability
3. **Pick thresholding strategy**
   - If using a global threshold, validate cross-query calibration (the paper shows raw cosine thresholds can fail).
4. **Precompute offline trade-offs**
   - Use PR AUC, P@R95, Filter%, Null%, and ranking metrics like MRR where relevant.
5. **Segment analysis**
   - Head vs tail queries, misspellings, rare brands/terms, numeric queries
   - The paper notes recall loss tends to cluster in rare words and misspellings.
6. **Analyze “impacted queries”**
   - Sampling queries where ranking changed can help sensitivity for relevance judgments, but ensure this is understood as a *conditional* analysis (not necessarily representative of all traffic).

Related pages: [[query_segmentation]], [[long_tail_queries]], [[latency_budget]]

---

## Notes on methodology relevance to experimentation

Although primarily a modeling contribution, the Cosine Adapter paper contributes experiment-design-relevant ideas:

- **Calibration enables safer global policies**
  - Mapping raw cosine similarity into a more interpretable relevance score supports consistent global thresholding.
- **Compute-aware design**
  - Filtering candidates before reranking can reduce unnecessary reranker load; experiments should track latency/cost.
- **Two-stage validation**
  - Offline benchmarks (MS MARCO + internal data) → then production A/B test.

---

## References (source integrated)

- Rossi, N., Lin, J., Liu, F., Yang, Z., Lee, T., Magnani, A., & Liao, C. (2024). *Relevance Filtering for Embedding-based Retrieval*. CIKM ’24. https://doi.org/10.1145/3627673.3680095  
  Resources: https://github.com/juexinlin/dense_retrieval_relevance_filter