---
slug: deep_learning_fundamentals
sources:
- blog.reachsumit.com
tags: []
title: Deep Learning Fundamentals
updated: '2026-04-14'
---

# Deep Learning Fundamentals

This page introduces core deep learning concepts through the lens of *industrial-scale retrieval and ranking*, focusing on representation learning with **Two-Tower (dual-encoder / bi-encoder)** architectures and their efficiency-driven extensions.

> Related: [[information_retrieval]], [[recommender_systems]], [[ranking_and_retrieval]], [[contrastive_learning]], [[transformers]], [[bert]]

---

## Why deep learning shows up in retrieval and ranking systems

Large-scale search and recommendation systems often have **tens of millions** of candidate items/documents. A single complex deep model is typically too slow to score everything end-to-end under strict latency constraints.

### Cascade (multi-stage) ranking systems

A common production design is a **multi-stage (cascade) ranking pipeline** that trades off *efficiency* and *effectiveness*:

- **Recall / Retrieval stage**
  - Fast, high-recall candidate generation from a huge corpus.
- **Pre-ranking stage**
  - Scores a *larger* candidate set than the final ranker must handle.
  - Prioritizes *inference efficiency* while improving quality over pure retrieval.
  - Two-tower models are commonly used here.
- **Ranking / Re-ranking stages**
  - More expensive, interaction-heavy deep models applied to fewer candidates.
  - Focus on precision/utility metrics.

**Latency matters:** reported findings suggest that even ~100ms increases can measurably degrade user experience and revenue.

> Related: [[latency_optimization]], [[approximate_nearest_neighbors]], [[vector_search]]

---

## Two-Tower (Dual Encoder / Bi-Encoder) models

### Core idea

A **Two-Tower** model (also called **dual encoder** or **bi-encoder**) computes two embeddings independently and combines them late:

- **User/query tower**: maps user features or query text to an embedding vector
- **Item/ad/document tower**: maps item features or document text to an embedding vector
- **Similarity / score**: typically **inner product** (dot product) between the two vectors

Key property: **late interaction**  
The two towers do not interact until the final similarity computation.

### Why Two-Tower models are popular in production

Two-tower architectures are widely adopted in:
- content recommendation,
- advertising systems,
- search engines,

and are described as a go-to solution for **pre-ranking** because they balance:

- **Accuracy**: learn dense latent representations from rich features
- **Efficiency**:
  - Towers run **independently in parallel**
  - Item/document embeddings can be **precomputed**, **frozen**, and stored in an index for fast inference
  - New/updated items can be added to the index without retraining the whole model (common in dense retrieval setups)

> Related: [[embedding_models]], [[nearest_neighbor_search]], [[dense_retrieval]]

### Historical note (ads pre-ranking evolution example)

One described evolution path for ad pre-ranking:
1. Non-personalized score via smoothed/averaged recent CTR
2. Logistic regression (lightweight; supports online learning/serving)
3. Two-tower DNN producing user and ad vectors; dot product yields pre-rank score

---

## Related neural ranking paradigms (representation vs interaction)

In neural matching (e.g., query → document), two-tower models are **representation-based**. Other paradigms introduce stronger interactions:

- **Representation-based (Two-Tower)**
  - Independent embeddings; interaction only at output
- **Interaction-based models (e.g., DRMM, KNRM)**
  - Build an interaction matrix (word/phrase-level) and feed to CNN/MLP
- **Cross-encoders (e.g., BERT as cross-encoder)**
  - Jointly encode query+document with full attention; high accuracy, high cost
- **Late-interaction hybrids (e.g., ColBERT)**
  - Preserve *query-document decoupling* but allow richer matching at the output using token-level representations

Practical takeaway:
- Two-tower/dual-encoder architectures are preferred when you need **fast retrieval or pre-ranking at scale**.
- Cross-encoders are preferred when you can afford scoring far fewer candidates.

> Related: [[colbert]], [[cross_encoder]], [[bi_encoder]], [[semantic_search]]

---

## Dual encoder architectural variants

Dual encoders can be structured in multiple ways:

### Siamese Dual Encoder (SDE)

- Two identical sub-networks
- Often share parameters
- Historically proposed for tasks like signature verification (Siamese networks)

### Asymmetric Dual Encoder (ADE)

- Two towers have distinct parameters
- Used when asymmetry is needed between inputs

#### Empirical note on SDE vs ADE (question-answer retrieval study)

A reported study found:
- **SDEs outperform ADEs** because ADEs can embed inputs into **disjoint embedding spaces**, hurting retrieval quality.
- ADE performance can be improved by **sharing a projection layer** (**ADE-SPL**), sometimes matching or exceeding SDE.
- Sharing token embedders (**ADE-STE**) or freezing token embedders (**ADE-FTE**) yields only marginal improvements.

> Related: [[metric_learning]], [[representation_learning]]

---

## A key limitation of classic Two-Tower models

Because the towers learn representations largely **independently**, classic two-tower designs can suffer from:

- **Insufficient feature interaction** between user/query and item/document representations
- Reduced ability to model fine-grained matching signals compared to interaction-heavy models

This motivates extensions that add interaction while preserving efficiency.

---

## Enhancements and extensions to Two-Tower models

### Dual Augmented Two-Tower Model (DAT)

Goal: introduce cross-tower interaction information *without fully coupling inference*.

Approach:
- Augment each tower’s input with a vector that captures **historical positive interaction information from the other tower**
  - e.g., add vectors \(a_u\) and \(a_v\) to user and item tower inputs
- These augmentation vectors are updated during training to model cross-tower information.
- Adds a **category alignment loss** to address category imbalance by transferring knowledge from data-rich categories to sparse ones.

Caveat noted in the source:
- Later research reports the gains of DAT are still **limited**.

> Related: [[class_imbalance]], [[loss_functions]]

### Interaction Enhanced Two-Tower Model (IntTower)

Stated design objective:
- Increase **information interaction** while keeping **inference efficiency** acceptable for pre-ranking.

IntTower introduces three components:

#### 1) Light-SE Block (feature importance / recalibration)

- Inspired by **Squeeze-and-Excitation Networks (SENet)**
- Purpose:
  - Identify importance of different features
  - Produce refined feature representations in each tower
- Compared with SENet:
  - Uses a more **lightweight** design (e.g., a single FC layer) to compute feature importance.

> Related: [[attention_mechanisms]], [[feature_engineering]]

#### 2) FE-Block (Fine-grained and Early Feature Interaction)

- Inspired by ColBERT’s late interaction style
- Performs **fine-grained early feature interaction** between:
  - multi-layer user representations, and
  - the last layer of item representation

#### 3) CIR Module (Contrastive Interaction Regularization)

- Uses an **InfoNCE-style contrastive loss** to bring user embeddings closer to positive items
- Training objective combines:
  - traditional supervised loss (e.g., log loss between predicted scores and labels)
  - plus the contrastive regularization loss

> Related: [[infonce]], [[contrastive_learning]], [[supervised_learning]]

#### Reported results and operational considerations

- IntTower reportedly outperforms several pre-ranking baselines:
  - Logistic Regression, Two-Tower, DAT, COLD
- It reportedly performs comparably to some ranking models:
  - Wide&Deep, DeepFM, DCN, AutoInt
- Efficiency notes:
  - Parameter count and training time increases are described as **negligible**
  - Serving latency is described as **acceptable**
- Representation quality:
  - t-SNE visualizations suggest users and positive items cluster more tightly than in classic two-tower models; negatives are farther away.

---

## Alternatives to Two-Tower for pre-ranking

Some research explores **single-tower** structures to fully model interactions, but these often lose the key production advantage of two-tower systems:

- Without **user-item decoupling**, inference can become significantly more expensive.
- Systems may rely on optimization/approximation tricks to preserve serving performance.

Examples mentioned:

- **COLD**: “computing power cost-aware online and lightweight deep pre-ranking system”
  - Uses offline feature selection to optimize QPS and response time vs quality.
- **FSCD**: feature selection with learnable dropout/regularization
  - Attempts to inherit effective pre-ranking behavior from a stronger ranking model via feature-wise regularization.

> Related: [[model_compression]], [[feature_selection]], [[serving_systems]]

---

## Practical takeaways (fundamentals emphasized by the source)

- Deep learning in retrieval/ranking is often about **representation learning under latency constraints**.
- Two-tower models are foundational because they support:
  - scalable indexing,
  - decoupled inference,
  - fast similarity computation.
- Modern extensions (DAT, IntTower, late-interaction models like ColBERT) attempt to mitigate the classic **lack of interaction** while retaining efficiency.

---

## Contradictions / tensions to note

Because this page is newly created from a single source, there are **no direct contradictions with prior page content** yet.

However, the source itself highlights an important *design tension* (not a strict contradiction, but a tradeoff that often leads to conflicting choices in practice):

- **Two-Tower strength:** efficiency and decoupling (fast indexing and inference)
- **Two-Tower weakness:** limited cross-input interaction
- **Cross-encoders strength:** rich interaction modeling and often higher accuracy
- **Cross-encoders weakness:** much higher inference cost (hard to apply at large candidate sizes)

---

## References (from source)

- Deep Structured Semantic Model (DSSM): Huang et al. (2013)
- ColBERT: Khattab & Zaharia (2020)
- Exploring Dual Encoder Architectures for QA: Dong et al. (2022)
- Siamese networks: Bromley et al. (1993)
- DAT: Yu et al. (2021)
- IntTower: Li et al. (2022)
- Squeeze-and-Excitation Networks (SENet): Hu et al. (2017)
- InfoNCE overview resource: Lilian Weng (blog reference)
- COLD: Wang et al. (2020)
- FSCD: Ma et al. (2021)