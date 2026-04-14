---
slug: ranking_metrics
sources:
- hav4ik.github.io
tags: []
title: Ranking Metrics and Evaluation
updated: '2026-04-14'
---

# Ranking Metrics and Evaluation

This page collects the core *ranking quality metrics* used in information retrieval (IR) and Learning-to-Rank (LTR), and explains how they are used for **offline evaluation** (with human labels) and **counterfactual/online evaluation** (with interaction data like clicks).

Related pages to cross-reference:
- Learning-to-Rank: [[learning_to_rank]]
- Unbiased / Counterfactual LTR and click models: [[unbiased_learning_to_rank]]
- Online evaluation and A/B testing: [[online_evaluation]]
- Retrieval vs re-ranking architecture context: [[search_architecture]]

---

## 1) What “ranking evaluation” measures

Given a query and a set of retrieved documents, a ranker outputs an ordering (often via per-document scores). Evaluation asks: **does the ordering put the most useful/relevant items near the top?**

In web search and recommender/search stacks, evaluation is typically aligned to a mixture of relevance definitions:

- **Human-labeled relevance** (graded labels or pairwise preferences)
  - Commonly: discrete grades (e.g., 0–4 or 1–5) assigned under strict guidelines.
- **Implicit feedback** (behavioral signals)
  - **Click-through rate (CTR)** as a proxy for relevance (cheap but biased; see biases below).
- **Business outcomes**
  - **Conversion rate** (purchases, signups, profit, etc.; definition depends on product).

> Note: These are *signals*, not equivalent ground truth. CTR and conversions can be heavily confounded by presentation and selection effects.

---

## 2) Offline ranking metrics (label-based)

Offline metrics are computed from judged relevance labels (binary or graded). Common IR/LTR metrics include:

- **MAP (Mean Average Precision)**
- **MRR (Mean Reciprocal Rank)**
- **ERR (Expected Reciprocal Rank)**
- **NDCG (Normalized Discounted Cumulative Gain)**

### Important nuance: MAP/MRR vs graded relevance

New source emphasizes:

- **MAP and MRR are widely used for document retrieval**, but **often not preferred for search result ranking** when you have *graded relevance*, because they **do not incorporate per-document relevance grades** (they are naturally binary relevance metrics).
- You *can* still use them with:
  - binary labels, or
  - modified versions that account for graded relevance (implementation-dependent).

This is mostly consistent with common practice: NDCG and ERR are preferred when labels are graded and position sensitivity matters.

---

## 3) NDCG (Normalized Discounted Cumulative Gain)

NDCG is one of the most common metrics for ranked lists with **graded relevance** and **position sensitivity**.

### 3.1) DCG definition

For a query, the **Discounted Cumulative Gain** at truncation level \(T\) is:

\[
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
\]

Where:
- \(T\) = truncation level (e.g., 10 for “first page” evaluation)
- \(l_i\) = relevance label/grade for the item at rank \(i\) (e.g., 0–4, or 1–5)

**Interpretation**
- \(2^{l_i}-1\) is a **gain** function: higher grades contribute exponentially more.
- \(\log(1+i)\) is a **discount** function: lower ranks contribute less.

### 3.2) NDCG normalization

NDCG normalizes DCG by the *maximum achievable DCG* for that query:

\[
NDCG@T = \frac{DCG@T}{\max DCG@T}
\]

So:
- \(NDCG@T \in [0, 1]\)
- \(1\) means a perfect ordering (at least up to truncation \(T\))

**Practical notes**
- Because of normalization, NDCG is more comparable across queries with different label distributions.
- A “bad” ranking can still have non-zero NDCG if it contains some relevant documents in the top \(T\).

---

## 4) ERR (Expected Reciprocal Rank)

ERR models a user who scans results top-down and stops when they find a satisfying result.

The metric is:

\[
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right),
\quad \text{where} \quad
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
\]

Where:
- \(R_i\) represents the probability the user finds the result at rank \(i\) relevant/satisfying
- \(l_m\) is the maximum possible label value

**Interpretation**
- High-grade results early increase ERR substantially.
- The product term \(\prod_{i<r}(1-R_i)\) captures the probability the user has *not* already been satisfied earlier.

---

## 5) Truncation and “@K” reporting

Most ranking metrics are reported with a cutoff:

- **NDCG@1, NDCG@3, NDCG@5, NDCG@10** are common in web search.
- Truncation aligns the metric with how users interact (most attention is on top results).

This also aligns with training objectives in many LTR systems, where optimization focuses on the top of the ranking.

---

## 6) Why ranking metrics are hard to optimize directly

Many ranking metrics depend on **sorting**, which is non-differentiable:

- Metrics like **NDCG** and **ERR** are **not differentiable** because ranks change discontinuously when scores swap order.
- This motivates surrogate losses or special approaches:
  - pairwise/listwise surrogates (e.g., RankNet, ListNet)
  - gradient “shaping” methods like LambdaRank/LambdaMART
  - probabilistic frameworks like LambdaLoss

Cross-reference: [[learning_to_rank]]

---

## 7) Counterfactual and online evaluation (interaction-based)

When evaluation uses **click logs** (or other interactions) from a deployed ranker (“policy”), naïvely computing metrics on clicks is misleading due to bias.

### 7.1) Common click signal biases

The source highlights three major biases:

- **Position bias**
  - Users examine and click higher-ranked items more, regardless of true relevance.
  - Design changes (layout, snippets, UI) can change examination patterns; estimators must be revisited after major redesigns.
- **Selection bias**
  - Some items have *zero probability* of being examined (e.g., results on page 2+).
  - Important distinction: some methods can correct position bias only if selection bias is not present (or is handled explicitly).
- **Trust bias**
  - Users may trust the system and click top-ranked items more, even if not relevant.

Cross-reference: [[unbiased_learning_to_rank]]

### 7.2) Counterfactual evaluation (policy evaluation)

Counterfactual evaluation estimates the quality of a **new ranking function** \(f_\theta\) using interaction data logged under an **old/deployed** ranker \(f_{\text{deploy}}\).

Terminology often used:
- Deployed ranker = **behavior policy**
- New candidate ranker = **evaluation policy**

This is foundational for:
- offline evaluation using logs (before risking online traffic)
- unbiased learning-to-rank methods that reweight or model exposure

Cross-reference: [[unbiased_learning_to_rank]], [[online_evaluation]]

---

## 8) Using metrics in practice (reporting and model development)

Common patterns in production/benchmarking:

- Report **multiple cutoffs** (e.g., NDCG@1/3/5/10) to reflect different user attention regimes.
- Prefer **NDCG/ERR** for graded labels and top-heavy utility.
- Use **MAP/MRR** primarily when relevance is binary or when the task is “find the first relevant item” (MRR is common in QA-style retrieval).

### Example: NDCG in training/evaluation loops

In LambdaMART/LightGBM-style training, evaluation often logs NDCG at several cutoffs during boosting rounds (e.g., NDCG@1,3,5,10) to track ranking quality during learning.

Cross-reference: [[learning_to_rank]]

---

## 9) Contradictions / cautions from the source

Because this page is new, there is no existing content to contradict. However, the new source includes a claim worth flagging as *potentially inconsistent with broader common practice*:

- The source states: **“MAP and MRR are widely used for documents retrieval but not for search results ranking.”**
  - **Caution:** In many IR settings, “document retrieval” and “search results ranking” are not cleanly separable—retrieval systems *do* rank results. In practice, MAP/MRR are still widely used for some ranking tasks (especially with binary relevance, navigational queries, QA, or passage retrieval benchmarks).
  - A more precise interpretation is: **for web search with graded relevance and strong position sensitivity, NDCG/ERR are often preferred**.

This page adopts that clarified interpretation while preserving the source’s intent.

---

## 10) Glossary

- **Query**: user input; may include context in behavioral settings.
- **Document/item**: candidate result being ranked.
- **Relevance label (graded)**: discrete score representing judged relevance (e.g., 0–4).
- **Truncation level (@T, @K)**: metric computed only over top \(T\) results.
- **Gain**: how much credit a relevance label contributes (e.g., \(2^{l}-1\)).
- **Discount**: reduces credit at lower ranks (e.g., \(\log(1+i)\)).
- **Behavior policy / evaluation policy**: deployed ranker vs candidate ranker in counterfactual evaluation.

---