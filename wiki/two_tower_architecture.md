---
slug: two_tower_architecture
sources:
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
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

**New (CIKM’24) nuance:** in many embedding-based retrieval systems, the similarity used at retrieval time is specifically **cosine similarity** between normalized embeddings, and the resulting cosine scores are often **not directly interpretable as “absolute relevance”** across different queries (see **Relevance filtering & score calibration** below).

---

## Where it fits in cascade (multi-stage) ranking systems

Industrial retrieval systems often operate over **tens of millions** of candidates and must meet strict latency constraints (e.g., user experience and revenue can measurably degrade with ~100ms additional latency). As a result, a **multi-stage (cascade) ranking system** is commonly used:

- **Recall / Retrieval stage**: very fast candidate generation optimized for recall.
- **Pre-ranking stage**: filters candidates further with a lightweight model that can score many items quickly.
- **Ranking / Re-ranking stages**: more expensive, interaction-heavy models applied to far fewer candidates.

Two-tower models are described as a common go-to choice for **pre-ranking**, balancing:
- **Efficiency** (fast scoring, index-friendly embeddings)
- **Effectiveness** (learned semantic similarity rather than hand-engineered matching)

**Extension from new source:** even when retrieval is high-recall, **precision control** can become a bottleneck—especially in domains like **e-commerce product search**, where the number of truly relevant products for a query may be small. Retrieving top-*K* with ANN may include many irrelevant results, increasing downstream **re-ranker** load and potentially harming user experience. This motivates adding a lightweight **relevance filtering** step between ANN retrieval and reranking (see below).

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

**New source detail (dense retrieval background):**
- In embedding-based retrieval, a common workflow is:
  1. Train dual encoders to embed query and candidate to `R^d`
  2. Precompute candidate embeddings and build an ANN index
  3. At query time, compute query embedding and run ANN to fetch **top-*K*** by similarity (often cosine)
- Common dual-encoder training objectives include:
  - **Contrastive loss** with in-batch negatives (temperature-scaled softmax over positives vs negatives)
  - **Softmax listwise loss** using graded labels over a candidate set

Related: [[dense_retrieval]], [[approximate_nearest_neighbor_search]].

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

**New source note:** the relevance filtering paper depicts its “Cosine Adapter” on top of a **Siamese dual encoder** and explicitly **freezes the dual encoder** while training the adapter.

Related: [[siamese_networks]], [[metric_learning]].

---

## Limitations of the classic Two-Tower model

A commonly cited limitation is **insufficient interaction** between the two sides:

- Each tower learns its embedding **independently**, without “seeing” the other tower’s features during representation learning.
- This can reduce the model’s ability to capture fine-grained feature interactions that matter for relevance, especially compared to interaction-heavy or cross-encoder models.

This limitation motivates many proposed extensions.

---

## New: Relevance filtering & score calibration for embedding-based retrieval

### Motivation: why “top-*K*” and raw cosine cutoffs can be insufficient

The new source (Rossi et al., CIKM’24) highlights a production issue in dense retrieval:

- Dense retrieval via ANN has **no natural cutoff** like lexical retrieval (keyword matching inherently limits candidate set size).
- Two-tower similarity scores (often **cosine similarity**) are typically optimized by **contrastive** or **ranking** losses that enforce **relative ordering**, not absolute relevance.
- Therefore:
  - Raw cosine scores can be **hard to interpret**
  - Cosine scores are often **not comparable across different queries**
  - A global “cosine threshold” or “top-*K*” rule may produce poor **precision**, especially when the number of relevant items is small (common in product search)

This is closely related to broader topics like **score calibration** and **ranked list truncation**, but the proposed method differs from “predict a cutoff position” approaches.

Related: [[dense_retrieval]], [[approximate_nearest_neighbor_search]].

### Cosine Adapter: a lightweight query-dependent mapping from cosine to calibrated relevance

**Idea:** add a small neural module that maps raw cosine similarity to an interpretable, query-comparable relevance score.

- Let `x = cos(q, p)` be the cosine similarity between query embedding and candidate embedding.
- Learn a **monotonic transformation** `F_Θ(x)` where parameters `Θ` are **query-dependent**:
  - A small feedforward network (“Cosine Adapter”) takes the **query embedding** as input and outputs `Θ`
  - Apply `F_Θ` to each candidate’s cosine score
  - Then apply a **global threshold** `t` on the calibrated scores to filter candidates:
    - `p` is kept if `F_Θ(cos(q,p)) ≥ t`

**Why query-dependent?**
- It can adapt to “query difficulty” and other query-specific properties, enabling cross-query comparability of calibrated scores.

**Example transformation families explored (all monotonic, chosen to preserve ranking as much as possible):**
- Raw: `F(x)=x` (baseline)
- Linear: `F(x|a,b)=ax+b`
- Square root: `F(x|a,b)=sgn(x)·a·sqrt(|x|)+b`
- Quadratic: `F(x|a,b)=sgn(x)·a·x^2+b`
- Power: `F(x|a,b,k)=sgn(x)·a·|x|^k+b`, with `k∈(0,2)`

**Training setup from the source:**
- Freeze the dual encoder; train only the adapter layers.
- Optimize a **binary cross-entropy** objective, treating `F_Θ(x)` as a logit and `σ(F_Θ(x))` as a probability of relevance.

**Operational placement in the pipeline:**
1. Encode query
2. ANN retrieve top-*K* candidates + raw cosine scores
3. Run Cosine Adapter once per query to get `Θ`
4. Calibrate each candidate score (O(*K*) simple operations)
5. Apply a global threshold; send filtered set to reranker

**Complexity notes (from source):**
- Adapter forward pass once per query: about `O(d^2)` for embedding dimension `d` (small compared to reranking)
- Calibrating `K` candidates: `O(K)`
- Contrasts with some ranked list truncation methods using self-attention over candidates, which can be `O(K^2 d)`

Related: [[ranking]], [[pre_ranking]], [[probability_calibration]] (if present), [[ranked_list_truncation]] (if present).

### Reported effects and trade-offs

The relevance filtering approach is explicitly a **precision–recall trade-off**:

- Goal: **increase precision** of retrieved set with **small recall loss**
- Measured with metrics such as:
  - PR AUC
  - Precision at high recall target (e.g., **P@R95**)
  - Filter% (fraction filtered out)
  - Null% (queries returning zero results after filtering)

**Empirical highlights (from the source):**
- On MS MARCO passage retrieval, calibrated mappings improve PR AUC and filtering behavior relative to raw-score global thresholds, especially for large `K` (e.g., 1000).
- Raw-score global thresholds can lead to pathological behavior (spikes at “filter everything” or “filter nothing”), consistent with “scores not comparable across queries.”
- In Walmart product search experiments, calibrated mappings substantially improve PR AUC and P@R95 vs raw-score thresholds; best mapping shape can depend on whether the dual encoder was trained with contrastive vs listwise loss.
- In a production Walmart A/B test (with threshold tuned for very high recall, e.g., 99%), the method improved judged precision on impacted queries while showing neutral engagement effects (orders, GMV).

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

**Related adjacent line of work (new source): ranked list truncation**
- “Ranked list truncation” methods aim to decide **where to truncate** a ranked list (often using score patterns and/or document statistics).
- Some recent works use self-attention over the retrieved list to predict cutoff position (e.g., Choppy).
- The Cosine Adapter approach differs by using the **query embedding** to produce a **query-dependent score calibration**, then applying a single global threshold.

Related: [[feature_selection]], [[ranking]], [[ranked_list_truncation]] (if present).

---

## Practical notes and common usage patterns

- **Dot-product scoring** is common because it is fast and works naturally with ANN indices.
- **Cosine similarity** is also common in dense retrieval (especially when embeddings are normalized); however raw cosine values may be poorly calibrated across queries depending on training objective.
- **Freezing item embeddings** is a frequent serving optimization; items can be updated in the index without retraining the towers (depending on pipeline and embedding compatibility).
- Two-tower models are frequently used in:
  - content recommendations
  - advertisement retrieval and pre-ranking
  - web search retrieval and candidate generation
  - e-commerce product retrieval (often hybrid with lexical retrieval)

**New operational pattern (from relevance filtering work):**
- Add a lightweight, query-time component after ANN retrieval to **filter candidates before reranking** using a global threshold on **calibrated** scores (improves precision and reduces downstream cost).

---

## Contradictions / tensions to be aware of

- **Existing page emphasis:** dual encoders are “easy to productionize” because new/updated documents can be dynamically added to the embedding index without retraining.
- **Existing page note:** item tower embeddings are often frozen after training and indexed.

These statements are compatible in spirit, but there is an operational tension:

- **If embeddings are produced by a fixed (frozen) encoder**, you *can* add new documents/items by running them through the existing encoder and indexing the resulting embeddings (no retraining needed).
- **If the encoder itself is retrained**, previously indexed embeddings may become stale and need re-encoding/reindexing for consistency.

So “no retraining needed to add items” applies when the serving encoder is stable; frequent retraining introduces reindexing costs.

**Additional tension surfaced by the new source (score interpretability):**
- **Classic two-tower retrieval practice** often treats similarity scores (dot product / cosine) as if a global cutoff/top-*K* is meaningful.
- **New source claim:** cosine similarity scores learned with contrastive/listwise objectives are **not comparable across queries**, making a global cosine threshold “not optimal,” and top-*K* lacks a “natural cutoff” in dense retrieval.

These are not strictly contradictory, but they highlight that:
- Two-tower similarity is excellent for **ranking candidates within a query**, yet can be weak as an **absolute relevance signal** for deciding *how many* candidates to keep across diverse queries.
- This motivates post-retrieval calibration/filtering (e.g., Cosine Adapter) or other truncation methods.

---

## References (from source)

- Huang et al. (2013). DSSM: Deep Structured Semantic Models for web search.
- Bromley et al. (1993). Siamese network for signature verification.
- Khattab & Zaharia (2020). ColBERT (late interaction).
- Yu et al. (2021). Dual Augmented Two-tower Model (DAT).
- Li et al. (2022). IntTower.
- Hu et al. (2017). Squeeze-and-Excitation Networks (SENet).
- Dong et al. (2022). Dual encoder architecture study for QA retrieval.
- Plus pre-ranking systems and feature selection approaches: