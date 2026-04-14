---
slug: two_tower_architecture
sources:
- blog.reachsumit.com
tags: []
title: Two-Tower Architecture
updated: '2026-04-14'
---

# Two-Tower Architecture

**Two-Tower Architecture** (also called the **Two-Tower model**, **dual encoder**, or **bi-encoder**) is a representation-based deep neural network (DNN) paradigm widely used in large-scale **information retrieval** and **recommendation** systems—especially for **retrieval**, **pre-ranking**, and other early-stage ranking tasks where low latency and high recall are critical.

In a two-tower design, two separate neural networks (“towers” / “encoders”) independently embed the two sides of a matching problem—most commonly:
- **User ↔ Item** (recommendations / ads)
- **Query ↔ Document** (search / neural matching)
- **Question ↔ Passage** (dense retrieval / QA)

The system scores a pair by applying a simple similarity function (often an **inner product**) between the two embeddings. This enables fast approximate nearest neighbor (ANN) retrieval by precomputing and indexing one side’s embeddings (typically items/documents).

---

## Where it fits in cascade (multi-stage) ranking systems

Industrial retrieval systems often operate over **tens of millions** of candidates and must meet strict latency constraints (e.g., user experience and revenue can measurably degrade with ~100ms additional latency). As a result, a **multi-stage (cascade) ranking system** is commonly used:

- **Recall / Retrieval stage**: very fast candidate generation optimized for recall.
- **Pre-ranking stage**: filters candidates further with a lightweight model that can score many items quickly.
- **Ranking / Re-ranking stages**: more expensive, interaction-heavy models applied to far fewer candidates.

Two-tower models are described as a common go-to choice for **pre-ranking**, balancing:
- **Efficiency** (fast scoring, index-friendly embeddings)
- **Effectiveness** (learned semantic similarity rather than hand-engineered matching)

Related: [[cascade_ranking_system]] (if present), [[information_retrieval]], [[recommender_systems]].

---

## Core idea and structure

A standard two-tower model contains:

- **User/query tower**
  - Input: raw features (e.g., user profile, context, query text)
  - Layers: embeddings + MLP/DNN stack
  - Output: latent vector representation (embedding), e.g. `u ∈ R^d`

- **Item/document tower**
  - Input: raw features (e.g., item metadata, ad features, document text)
  - Layers: embeddings + MLP/DNN stack
  - Output: latent vector representation, e.g. `v ∈ R^d`

- **Scoring (“late interaction”)**
  - Commonly: `score(u, v) = u · v` (inner product / dot product)
  - Sometimes: cosine similarity or other metric in embedding space

Key characteristic:
- The towers are **decoupled** and can run **in parallel**, interacting only at the output layer—often called **late interaction**.

---

## Why it is efficient in production

Two-tower systems are popular not only for accuracy but for an inference design that supports scalable serving:

- **Embedding precomputation**
  - Item/ad/document embeddings can be computed offline and stored in an indexing service.
  - At inference, only the user/query embedding must be computed online.

- **ANN-friendly retrieval**
  - Once you have an embedding for the user/query, you can retrieve top candidates via nearest-neighbor search over item embeddings.

- **Frozen towers**
  - Common approach: freeze the **item tower** embeddings after training and index them for fast retrieval.
  - Some systems may even freeze **both towers** and update them periodically offline (e.g., retrain daily and rebuild indices).

Related: [[approximate_nearest_neighbor_search]], [[embedding_index]].

---

## Terminology and related architectures

Two-tower models sit within a broader family of neural matching / ranking architectures:

- **Representation-based rankers (Two-Tower / Dual Encoder)**
  - Encode each side independently; interaction happens only via a similarity function at the end.
  - Strength: scalable retrieval via indexing.
  - Weakness: limited cross-input interaction during modeling.

- **Interaction-based rankers**
  - Compute rich interaction features (e.g., token-to-token matrices) between query and document early, then score via CNN/MLP.
  - Examples mentioned in the source: **DRMM**, **KNRM**.

- **Cross-encoders (e.g., BERT cross-encoder)**
  - Jointly encode the concatenated query+document to model full interactions.
  - Powerful but much more expensive; typically used in later ranking stages.

- **Late-interaction hybrids (e.g., ColBERT)**
  - Preserve query-document decoupling for efficiency but incorporate more nuanced interaction at the output stage than a single dot product.

Related: [[bert]], [[colbert]], [[cross_encoder]], [[dense_retrieval]].

---

## Dual encoder variants: Siamese vs asymmetric

Two-tower models are often categorized by parameter sharing:

- **Siamese Dual Encoder (SDE)**
  - Two identical subnetworks (shared parameters).
  - Historically proposed for similarity tasks (e.g., signature verification).

- **Asymmetric Dual Encoder (ADE)**
  - Two towers have different parameters.
  - Used when the two sides require different inductive biases or feature processing.

Findings summarized in the source (from a study on QA retrieval dual-encoders):
- **SDEs can outperform ADEs**, because ADEs may embed inputs into **disjoint embedding spaces**, harming retrieval quality.
- ADE performance can be improved by **sharing a projection layer** (ADE-SPL).
- Sharing token embedders (ADE-STE) or freezing token embedders (ADE-FTE) yields only marginal gains in that study.

Related: [[siamese_networks]], [[metric_learning]].

---

## Limitations of the classic Two-Tower model

A commonly cited limitation is **insufficient interaction** between the two sides:

- Each tower learns its embedding **independently**, without “seeing” the other tower’s features during representation learning.
- This can reduce the model’s ability to capture fine-grained feature interactions that matter for relevance, especially compared to interaction-heavy or cross-encoder models.

This limitation motivates many proposed extensions.

---

## Promising extensions and improvements

### Dual Augmented Two-Tower Model (DAT)

**Goal:** introduce cross-tower feature interaction while largely preserving two-tower efficiency.

Approach (as described in the source):
- Augment the input of each tower with an additional vector capturing **historical positive interaction information** from the other side:
  - `a_u` and `a_v` get updated during training.
  - These vectors inject information about cross-tower interaction into each tower’s input.

Additional component:
- **Category alignment loss** to mitigate category imbalance by transferring knowledge from data-rich categories to others.

Note from the source:
- Later research suggests DAT’s performance gains can be **limited**.

Related: [[feature_interaction]], [[class_imbalance]].

---

### Interaction Enhanced Two Tower Model (IntTower)

**Goal:** improve modeling of interactions while keeping inference efficient for pre-ranking.

The IntTower proposal adds three components:

- **Light-SE Block**
  - Learns feature importance weights to refine feature representations in each tower.
  - Inspired by **Squeeze-and-Excitation Networks (SENet)**, but implemented more lightly (e.g., a single fully connected layer).

- **FE-Block (Fine-grained and Early Feature Interaction)**
  - Inspired by ColBERT-style interaction.
  - Performs fine-grained early interaction between **multi-layer user representations** and the **last layer item representation**.

- **CIR Module (Contrastive Interaction Regularization)**
  - Uses a contrastive objective (InfoNCE) to pull user embeddings closer to positive items and push away negatives.
  - Combined with standard logloss during training.

Reported outcomes in the source:
- IntTower outperforms several pre-ranking methods (e.g., Logistic Regression, Two-Tower, DAT, COLD).
- It can perform comparably to some heavier ranking models (Wide&Deep, DeepFM, DCN, AutoInt).
- Parameter and training-time increases are described as **negligible**, and serving latency as **acceptable**.

Related: [[contrastive_learning]], [[infonce]], [[senet]].

---

## Alternatives to Two-Tower for pre-ranking

Research also explores **single-tower** structures to fully model feature interactions and improve accuracy. However, these often lose the key “decoupling” benefit of two-tower architectures and therefore require efficiency optimizations.

Examples mentioned:

- **COLD (Computing power cost-aware Online and Lightweight Deep pre-ranking system)**
  - Uses offline feature selection to find feature sets that optimize effectiveness while meeting constraints like QPS and response time.

- **FSCD (Feature Selection based on feature Complexity and variational Dropout)**
  - Uses learnable dropout parameters for feature-wise regularization so the pre-ranking model can be effectively “inherited” from the ranking model.

Related: [[feature_selection]], [[ranking]].

---

## Practical notes and common usage patterns

- **Dot-product scoring** is common because it is fast and works naturally with ANN indices.
- **Freezing item embeddings** is a frequent serving optimization; items can be updated in the index without retraining the towers (depending on pipeline and embedding compatibility).
- Two-tower models are frequently used in:
  - content recommendations
  - advertisement retrieval and pre-ranking
  - web search retrieval and candidate generation

---

## Contradictions / tensions to be aware of

The source text emphasizes that dual encoders are “easy to productionize” because *new or updated documents can be dynamically added to the embedding index, without retraining the encoders*. It also notes that in practice *item tower embeddings are often frozen after training* and indexed.

These statements are compatible in spirit, but there is an operational tension:

- **If embeddings are produced by a fixed (frozen) encoder**, you *can* add new documents/items by running them through the existing encoder and indexing the resulting embeddings (no retraining needed).
- **If the encoder itself is retrained**, previously indexed embeddings may become stale and need re-encoding/reindexing for consistency.

So “no retraining needed to add items” applies when the serving encoder is stable; frequent retraining introduces reindexing costs.

---

## References (from source)

- Huang et al. (2013). DSSM: Deep Structured Semantic Models for web search.
- Bromley et al. (1993). Siamese network for signature verification.
- Khattab & Zaharia (2020). ColBERT (late interaction).
- Yu et al. (2021). Dual Augmented Two-tower Model (DAT).
- Li et al. (2022). IntTower.
- Hu et al. (2017). Squeeze-and-Excitation Networks (SENet).
- Dong et al. (2022). Dual encoder architecture study for QA retrieval.
- Plus pre-ranking systems and feature selection approaches: COLD, FSCD; and contrastive learning (InfoNCE).