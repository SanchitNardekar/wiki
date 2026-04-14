---
slug: recommendation_systems
sources:
- blog.reachsumit.com
tags: []
title: Recommendation Systems
updated: '2026-04-14'
---

# Recommendation Systems

Recommendation systems (recsys) are used to retrieve and rank items (e.g., content, ads, documents, products) for a user under strict **scale** (often tens of millions of candidates) and **latency** constraints. They are widely deployed in **content recommendations**, **advertising**, and **search engines**, and commonly rely on **multi-stage (“cascade”) ranking architectures** to balance effectiveness and efficiency.

## Where recommendation fits: cascade / multi-stage ranking systems

Large-scale retrieval and recommendation services often use a **multi-stage ranking pipeline** because a single complex model cannot score all candidates within latency budgets.

Typical stages:

- **Recall / Retrieval (high recall, low cost):**
  - Quickly narrows the universe to a manageable candidate set.
  - Often uses indexing and approximate nearest neighbor (ANN) lookup over embeddings.
- **Pre-ranking (fast DNN scoring on large candidate sets):**
  - Further filters and sorts candidates from retrieval.
  - Often uses architectures designed for **inference efficiency**, notably the **Two-Tower** / dual-encoder family.
- **Ranking / Re-ranking (expensive, high quality):**
  - Applies heavier models that incorporate richer feature interactions.

Latency sensitivity:

- Industry observations report that even ~**100ms** additional response time can measurably degrade user experience and revenue, motivating staged designs and lightweight early models.

## Two-Tower (Dual Encoder / Bi-Encoder) models in recommendation systems

A **Two-Tower model** is a representation-based deep model that computes:

- a **user/query embedding** from user/query features via an embedding layer + DNN (“user tower”)
- an **item/ad/document embedding** from item features via an embedding layer + DNN (“item tower”)
- a **similarity score** (commonly an **inner product**) between the two embeddings

Key properties:

- **Late interaction:** the towers are computed independently and only “interact” at the output (e.g., dot product).
- **Inference efficiency:** item embeddings can be **precomputed** and stored in an **indexing service**, enabling fast retrieval/scoring.
- **Common use:** described as a “go-to” approach for industrial-scale **pre-ranking** tasks, and also used broadly in retrieval and matching.

Cross-references:

- Two-tower models are central to [[ranking]] and [[information_retrieval]] style systems and are frequently used in [[advertising_systems]] and [[search_engines]] (create these pages if missing).

## Related neural ranking paradigms (interaction vs representation)

In neural matching (e.g., query→document), Two-Tower is one point on a spectrum:

- **Representation-based rankers (Two-Tower):**
  - Encode query and document independently; compute similarity at the end.
  - Efficient due to embedding decoupling and indexing.
- **Interaction-based rankers (e.g., DRMM, KNRM):**
  - Build interaction matrices across tokens/phrases, then apply CNN/MLP-like modules.
  - More expressive but typically higher inference cost.
- **Cross-encoders (e.g., BERT-style):**
  - Jointly encode query and document with full attention across both.
  - Very strong quality, but expensive at serving time.
- **Late-interaction hybrids (e.g., ColBERT):**
  - Preserve *query–document decoupling* while allowing richer internal interactions.
  - Can freeze and index document embeddings, retaining some efficiency benefits.

These families often map directly onto recommender settings as:

- query ↔ user
- document ↔ item

## Dual encoder architecture variants

Two-tower models are also called **dual encoders** / **bi-encoders**. They appear in multiple IR tasks (question answering, entity linking, dense retrieval) and are considered easier to productionize because:

- new/updated items can be embedded and added to the index **without retraining** the full model.

Common variants:

- **Siamese Dual Encoder (SDE):**
  - Two identical encoders (often weight sharing).
  - Originally introduced for similarity/distance learning (e.g., signature verification).
- **Asymmetric Dual Encoder (ADE):**
  - Two distinct encoders (different parameters) used when asymmetry is needed.
  - Reported issue: can embed inputs into disjoint spaces, reducing retrieval quality.
  - Improvement: sharing a **projection layer** between encoders (ADE-SPL) can bring performance close to or beyond SDE in some settings.
  - Sharing token embedders (ADE-STE) or freezing token embedders (ADE-FTE) yields only marginal gains (per the cited QA dual-encoder study).

## Limitations of Two-Tower models (and why extensions exist)

A frequently cited limitation:

- **Lack of rich feature interaction between towers.**
  - Because towers are trained to produce embeddings largely independently (with only late interaction), models may miss useful cross-features that require early or fine-grained user–item interaction.

This motivates “enhanced two-tower” architectures that try to increase interaction while preserving serving efficiency.

## Enhancements and extensions to Two-Tower models

### Dual Augmented Two-Tower (DAT)

Goal: introduce cross-tower information while keeping two-tower structure.

Approach:

- Augment each tower’s input with an additional vector capturing **historical positive interaction information** from the other side:
  - user tower input includes an item-derived interaction vector (and vice versa).
- Adds a **category alignment loss** to mitigate category imbalance by transferring knowledge from data-rich categories to sparse ones.

Note:

- Later research reportedly found DAT gains **limited**.

### Interaction Enhanced Two-Tower (IntTower)

Goal: improve **interaction modeling** while keeping **inference efficiency** acceptable for pre-ranking.

Introduces three main components:

- **Light-SE Block**
  - A lightweight feature-importance module inspired by SENet (“Squeeze-and-Excitation Networks”).
  - Intuition: recalibrate feature channels by emphasizing informative features and suppressing less useful ones.
  - IntTower uses a simplified (lighter) design (e.g., a single fully connected layer) to estimate feature importance.
- **FE-Block (Fine-grained and Early Feature Interaction)**
  - Inspired by ColBERT’s late interaction style.
  - Performs fine-grained early interaction between multi-layer user representations and the last layer of item representation.
- **CIR Module (Contrastive Interaction Regularization)**
  - Uses a contrastive objective (InfoNCE) to bring user embeddings closer to positive items and push away negatives.
  - Training loss combines this regularizer with standard supervised log loss on labels.

Reported outcomes (per the source):

- IntTower outperforms baseline pre-ranking methods (e.g., Logistic Regression, Two-Tower, DAT, COLD).
- Performs comparably to some heavier ranking-stage models (e.g., Wide&Deep, DeepFM, DCN, AutoInt) while keeping parameter and latency increases small/acceptable.
- Embedding visualizations (t-SNE) suggest better clustering of users with positive items vs negatives.

Cross-references:

- Contrastive objectives connect to [[contrastive_learning]] (create if missing).
- Feature interaction models connect to [[feature_interactions]] (create if missing).

## Alternatives to Two-Tower for pre-ranking

Some research explores **single-tower** structures to fully model feature interactions, but these lose the “user–item decoupling” property and can be less efficient at serving time. To mitigate efficiency degradation, such approaches rely on optimization tricks, for example:

- **COLD** (Computing power cost-aware Online and Lightweight Deep pre-ranking system)
  - Uses offline feature selection to find feature sets that balance QPS (queries per second) and response time (RT) with ranking quality.
- **FSCD**
  - Uses learnable dropout parameters for feature-wise regularization, effectively inheriting a pre-ranking model from a ranking model.

## Practical notes for system design

When choosing models for a recommendation pipeline:

- Use **decoupled embedding models** (Two-Tower/dual encoders) when:
  - you need fast candidate scoring over large sets
  - you can precompute and index item embeddings
  - latency budgets are tight
- Use more **interaction-heavy models** when:
  - candidate set is already small (later-stage ranking)
  - you can afford higher compute per request
  - quality gains justify cost

## Contradictions / tensions noted in the sources

Because this page is newly created, there is no prior content to contradict. However, the new source itself highlights a **design tension** (not a direct contradiction):

- Two-Tower models are presented as a **state-of-the-art “go-to”** solution for pre-ranking due to efficiency, **yet** their **limited cross-tower interaction** is a recognized weakness that motivates architectures like DAT and IntTower (which add interaction while trying to keep latency acceptable).

## References (from the integrated source)

- Kohavi et al. (2013). Online controlled experiments at large scale.
- Huang et al. (2013). DSSM: deep structured semantic models for web search using clickthrough data.
- Wang et al. (2020). COLD: Towards the Next Generation of Pre-Ranking System.
- Khattab & Zaharia (2020). ColBERT.
- Dong et al. (2022). Exploring Dual Encoder Architectures for Question Answering.
- Bromley et al. (1993). Siamese networks for signature verification.
- Yu et al. (2021). Dual Augmented Two-tower Model (DAT).
- Li et al. (2022). IntTower.
- Hu et al. (2017). Squeeze-and-Excitation Networks (SENet).
- Lilian Weng blog (InfoNCE overview).
- Ma et al. (2021). Learnable feature selection for pre-ranking (FSCD).