---
slug: ranking_metrics
sources:
- hav4ik.github.io
- relevance_filtering_for_embedding_based_retrieval.pdf
tags: []
title: Ranking Metrics and Evaluation
updated: '2026-04-14'
---

# Ranking Metrics and Evaluation

This page collects the core *ranking quality metrics* used in information retrieval (IR) and Learning-to-Rank (LTR), and explains how they are used for **offline evaluation** (with human labels) and **counterfactual/online evaluation** (with interaction data like clicks).

It also adds context from *metric learning / embedding retrieval* literature (Deep Metric Learning), where the system is often evaluated as a **retrieval + ranking-by-similarity** pipeline rather than a classic supervised “ranker over a candidate set”. This matters because the *evaluation metrics* used in embedding retrieval (e.g., Recall@K, Precision@K, mAP, rank-1) overlap with IR metrics, while the *training objectives* (contrastive / triplet / angular-margin losses) are very different from standard LTR losses.

**New (LLM reasoning / inference-time ranking) context:** some modern “reasoning” evaluations effectively turn *generation* into a *ranking / aggregation* problem by sampling multiple candidate outputs and selecting an answer via a voting/selection rule (e.g., **majority vote among K samples**). This introduces ranking-like evaluation questions (how to allocate a compute budget, how to aggregate candidates, and how sensitive the metric is to output length).

**New (embedding retrieval relevance filtering) context:** recent production work on dense / embedding-based retrieval highlights that *top-K retrieval* can have **no natural cutoff** (unlike lexical retrieval), and that **raw cosine similarity is often not comparable across queries**—making “filter by cosine threshold” unreliable. A proposed solution is *query-dependent score calibration* that maps cosine similarity into an interpretable relevance score so that a **global threshold** can be applied for filtering before downstream re-ranking.

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

### 1.1) Ranking in “embedding retrieval” (metric learning) settings

The new source (a Deep Metric Learning survey) describes systems where we learn an embedding model $f_\theta(x)$ and rank candidates by a distance/similarity $\mathcal{D}(f_\theta(q), f_\theta(d))$. This covers:
- image retrieval and reverse image search,
- product matching / dedup,
- face identification,
- nearest-neighbor retrieval in embedding space.

In these pipelines, “ranking evaluation” still means “put the best matches at the top”, but:
- labels are often **class identity** (“same instance/class?”) rather than graded topical relevance,
- evaluation is frequently **top-K retrieval** oriented (e.g., rank-1 accuracy, Recall@K, mAP),
- the ranking function may be **distance in embedding space** (e.g., $l_2$ or cosine similarity) rather than a learned scoring function over rich query-document features.

This is relevant to [[search_architecture]] because many modern systems use:
- a *retrieval stage* (ANN / vector search) and then
- possibly a *re-ranking stage* (LTR) on a smaller candidate set.

### 1.1.1) Dense retrieval has “no natural cutoff” (precision control problem)

The CIKM’24 “Relevance Filtering for Embedding-based Retrieval” source makes a key practical point for embedding-based retrieval:

- Lexical retrieval often has an *implicit cutoff* because keyword matching limits the retrieved set.
- Dense retrieval via ANN can always return **top-K** neighbors even when few (or zero) are truly relevant.
- If the number of relevant items is small (common in **product search**), maximizing recall can yield **low precision** and a poor user experience even if the top few are good.

Implication for evaluation and reporting:

- Standard top-K metrics (Recall@K, MRR@K) can look acceptable while the *tail of the retrieved set* contains many irrelevant items that still:
  - waste downstream re-ranker compute, and/or
  - leak into the final ranked list under failure modes.

Cross-reference: candidate-set sizing and stage boundaries in [[search_architecture]].

### 1.2) Ranking-like evaluation for sampled generation (LLM reasoning)

The new source (a post on improving DeepSeek-R1-distilled math reasoning models) highlights an evaluation pattern that is structurally similar to ranking/selection:

- For each problem (query), sample many candidate solutions (“rollouts”, “traces”) from a model.
- Evaluate:
  - **Pass@1**: fraction of problems where a *single sample* is correct (akin to “best-of-1”).
  - **Maj@K**: majority-vote accuracy after sampling $K$ candidates (common is **Maj@32**).

This resembles ranking evaluation because:
- you have a **set of candidates** and must apply an **aggregation/selection policy** (majority vote; or in other setups, reranking by score/reward),
- you often have a **budget** (tokens or number of samples) analogous to **@K truncation** in ranked lists,
- the metric can be **highly sensitive** to factors that change the distribution of candidates (prompting, decoding, maximum length, stopping rules).

Cross-reference (conceptual): aggregation policies for candidates can be viewed as an “architecture” choice similar to retrieval + reranking in [[search_architecture]] (generate candidates → select/aggregate).

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

For a query, the **Discounted Cumulative Gain** at truncation level $T$ is:

$$
DCG@T = \sum_{i=1}^T \frac{2^{l_i} - 1}{\log (1 + i)}
$$

Where:
- $T$ = truncation level (e.g., 10 for “first page” evaluation)
- $l_i$ = relevance label/grade for the item at rank $i$ (e.g., 0–4, or 1–5)

**Interpretation**
- $2^{l_i}-1$ is a **gain** function: higher grades contribute exponentially more.
- $\log(1+i)$ is a **discount** function: lower ranks contribute less.

### 3.2) NDCG normalization

NDCG normalizes DCG by the *maximum achievable DCG* for that query:

$$
NDCG@T = \frac{DCG@T}{\max DCG@T}
$$

So:
- $NDCG@T \in [0, 1]$
- $1$ means a perfect ordering (at least up to truncation $T$)

**Practical notes**
- Because of normalization, NDCG is more comparable across queries with different label distributions.
- A “bad” ranking can still have non-zero NDCG if it contains some relevant documents in the top $T$.

---

## 4) ERR (Expected Reciprocal Rank)

ERR models a user who scans results top-down and stops when they find a satisfying result.

The metric is:

$$
ERR = \sum_{r=1}^n \frac{1}{r} R_{r} \prod_{i=1}^{r-1} \left( 1 - R_i \right),
\quad \text{where} \quad
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
$$

Where:
- $R_i$ represents the probability the user finds the result at rank $i$ relevant/satisfying
- $l_m$ is the maximum possible label value

**Interpretation**
- High-grade results early increase ERR substantially.
- The product term $\prod_{i<r}(1-R_i)$ captures the probability the user has *not* already been satisfied earlier.

---

## 5) Truncation and “@K” reporting

Most ranking metrics are reported with a cutoff:

- **NDCG@1, NDCG@3, NDCG@5, NDCG@10** are common in web search.
- Truncation aligns the metric with how users interact (most attention is on top results).

This also aligns with training objectives in many LTR systems, where optimization focuses on the top of the ranking.

### 5.1) “@K” metrics in embedding retrieval / metric learning benchmarks

The new source highlights large-scale retrieval and identification benchmarks (e.g., face identification, landmark retrieval) that commonly report top-heavy metrics such as:
- **Rank-1 accuracy** (“is the correct match at position 1?”)
- **Recall@K** (“is at least one correct match in the top K?”)
- **mAP / MAP** (common in retrieval benchmarks; typically binary relevance at the class/instance level)

These metrics are conceptually compatible with IR ranking evaluation; the main difference is the underlying labeling scheme (often same/different identity) and candidate generation (often full-corpus nearest-neighbor).

> Cross-reference: embedding retrieval is often the *retrieval stage* in [[search_architecture]], and can be followed by LTR re-ranking [[learning_to_rank]].

### 5.1.1) New: ranked-list truncation and *filtering* as part of evaluation

The CIKM’24 dense retrieval source reframes an important nuance:

- In dense retrieval, choosing $K$ is not just an evaluation cutoff; it can be an *operational candidate-set size* sent to reranking.
- Because ANN always returns a top-K list, systems often need an explicit **relevance filtering** step that decides which retrieved items are “relevant enough” to keep.

This introduces additional evaluation questions beyond Recall@K:

- What is the **precision–recall tradeoff** of filtering *before reranking*?
- Does filtering create **null-result queries** (no items left)?
- Is the filtering rule stable across **queries of different difficulty**?

These issues connect to ranked-list truncation literature (predicting a per-query cutoff position) but in dense retrieval settings may be addressed via **score calibration** (see §10).

Cross-reference: retrieval → filtering → reranking as a pipeline design choice in [[search_architecture]].

### 5.2) “@K” analogs in sampled-generation evaluation (Pass@1, Maj@K)

The DeepSeek-R1 math source uses evaluation metrics that behave like “@K” cutoffs, but in **sample space** rather than document space:

- **Pass@1**: correctness of one draw (one rollout).
- **Maj@K (e.g., Maj@32)**: sample $K$ candidate answers and take the majority vote.

Important implications (ranking-metric-style “cutoff sensitivity”):
- Changing $K$ changes what you measure: *single-sample quality* vs *self-consistency under sampling*.
- There is a **compute budget** not only in number of samples $K$, but also in **token budget / max generation length** per sample.

---

## 6) Why ranking metrics are hard to optimize directly

Many ranking metrics depend on **sorting**, which is non-differentiable:

- Metrics like **NDCG** and **ERR** are **not differentiable** because ranks change discontinuously when scores swap order.
- This motivates surrogate losses or special approaches:
  - pairwise/listwise surrogates (e.g., RankNet, ListNet)
  - gradient “shaping” methods like LambdaRank/LambdaMART
  - probabilistic frameworks like LambdaLoss

Cross-reference: [[learning_to_rank]]

### 6.1) Parallel issue in metric learning: evaluation is ranking, training is a surrogate

Deep Metric Learning (DML) provides an analogous story:
- **Evaluation** is usually retrieval ranking (top-K, mAP, rank-1).
- **Training** is typically via surrogate objectives that enforce geometry in embedding space:
  - contrastive loss, triplet loss, and modern “angular margin” softmax variants.

This is not a contradiction with LTR; it’s the same general pattern (optimize a differentiable proxy for a ranking objective) but with different modeling assumptions (distance in embedding space vs feature-based scoring).

### 6.2) Parallel issue in RL-tuned generation: evaluation is aggregation over samples, training is a surrogate

The DeepSeek-R1 math source describes RL fine-tuning (GRPO and variants) where:
- **Evaluation** is often **Pass@1** and **Maj@K** under a fixed sampling/decoding setup.
- **Training** optimizes a surrogate objective (policy gradient with reward shaping/normalization, KL regularization, length penalties, etc.).

A key evaluation lesson from the source:
- Under some settings, **better Pass@1 does not imply better Maj@K** (they report “Bottom line: better Pass@1 does not mean better Maj@32!”).
- Therefore, the “metric you optimize mentally” during development must match the “metric you ship/compare on”, similar to how optimizing a pairwise loss doesn’t guarantee best NDCG unless aligned.

> Cross-reference: the “offline vs online” mismatch theme is also central in [[online_evaluation]] and [[unbiased_learning_to_rank]], though the DeepSeek source is about *model sampling/aggregation* rather than clicks.

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

Counterfactual evaluation estimates the quality of a **new ranking function** $f_\theta$ using interaction data logged under an **old/deployed** ranker $f_{\text{deploy}}$.

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

### 8.1) Metrics + architecture choices: retrieval vs re-ranking

From the metric learning source’s “case study” framing (large-scale retrieval competitions), a common real-world pattern is:

- **Stage A: embedding retrieval**
  - Train an embedding model $f_\theta$; retrieve candidates via nearest neighbors.
  - Evaluate with top-K retrieval metrics (Recall@K, rank-1, mAP).
- **Stage B: post-processing / re-ranking**
  - Apply query expansion, verification, or local feature matching (in vision), or other re-ranking logic.
  - In classic IR stacks, this role is often served by LTR re-rankers.

Cross-reference: [[search_architecture]], [[learning_to_rank]]

### 8.2) Metric hygiene for sampled-generation evaluations (decoding, stopping, budget)

The DeepSeek-R1 math source underscores that evaluation numbers can move substantially due to “evaluation protocol” choices, many of which are analogs of *logging policy* and *cutoff* choices in ranking:

- **Token budget / max_len matters**
  - They report that at *higher* token budgets (e.g., 32K vs 16K), **Maj@32 can get slightly worse**, attributing it to increased self-doubt/hallucination when the model has “more time to think”.
  - This is analogous to how changing **candidate set size** or **reranking depth** in [[search_architecture]] can change measured quality in non-monotonic ways.
- **Stop conditions matter**
  - They used a specific stop string for their tuned models because they