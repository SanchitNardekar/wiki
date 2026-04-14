---
slug: deep_learning_fundamentals
sources:
- blog.reachsumit.com
- hav4ik.github.io
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

## Deep metric learning (DML) fundamentals (new)

The two-tower retrieval framing is closely related to **deep metric learning**: learning an embedding space where “similar” entities are near and “dissimilar” entities are far.

- **Goal (supervised DML):**
  - Learn an embedding model \( f_\theta: \mathcal{X} \rightarrow \mathbb{R}^n \)
  - Choose a distance/similarity function \( \mathcal{D} \) (often fixed)
  - Ensure:
    - small \( \mathcal{D}(f_\theta(x_1), f_\theta(x_2)) \) when \(y_1=y_2\)
    - large \( \mathcal{D}(f_\theta(x_1), f_\theta(x_2)) \) when \(y_1\neq y_2\)

- **Common distances/similarities:**
  - Euclidean / \(l_2\) distance (classic metric learning formulation)
  - Cosine similarity / angular distance (common in angular-margin methods)

**Connection to retrieval systems:**
- Retrieval and recommendation with two-tower models can be seen as **metric learning at scale**:
  - learn embeddings (users/queries/items) + a similarity (dot product / cosine)
  - use an ANN index for fast nearest-neighbor search

> Related: [[metric_learning]], [[contrastive_learning]], [[representation_learning]], [[dense_retrieval]], [[vector_search]]

---

## Contrastive losses in metric learning (pairs, triplets, and beyond) (new)

Many embedding systems are trained by directly enforcing “pull positives together, push negatives apart”.

### Contrastive loss (pairwise)

A classic objective over pairs \((x_1, x_2)\) with margin \(\alpha\):

- If \(y_1=y_2\): minimize squared distance \( \mathcal{D}^2(f(x_1), f(x_2)) \)
- If \(y_1\neq y_2\): enforce a margin via \( \max(0, \alpha - \mathcal{D}^2(\cdot)) \)

Motivation for the margin:
- prevents a degenerate solution where all embeddings collapse to a single point.

> Related: [[loss_functions]], [[contrastive_learning]]

### Triplet loss

Uses triples \((x_a, x_p, x_n)\) with \(y_a=y_p\), \(y_a\neq y_n\):

\[
\mathcal{L}_\text{triplet} = \max(0, \mathcal{D}^2(a,p) - \mathcal{D}^2(a,n) + \alpha)
\]

Key practical ingredient: **negative mining**
- sample “hard” or “semi-hard” negatives where
  \[
  \mathcal{D}(a,n) < \mathcal{D}(a,p) + \alpha
  \]
- without mining, gradients can become sparse late in training (many triplets yield zero loss).

> Related: [[negative_sampling]], [[metric_learning]]

### Known issues with direct distance-based objectives (as summarized in the new source)

The new source highlights two recurring problems when optimizing directly in \(l_2\) space:

- **Expansion issue:** difficult to ensure all samples of a class collapse into a coherent global region (local constraints may not enforce global structure).
- **Sampling issue:** performance depends heavily on mining strategies, which becomes operationally awkward at scale (e.g., distributed training).

These issues help motivate objectives that behave more like classification with stronger geometric constraints (below).

---

## From softmax classification to discriminative embeddings (new)

The new source explicitly notes that **softmax cross-entropy** can be used for metric learning, but is often **inferior** to specialized metric-learning objectives in terms of producing tightly clustered, well-separated embeddings.

### Center loss (softmax + center regularizer)

Adds a term pulling embeddings toward learned class centers \(c_{y_i}\):

\[
\mathcal{L}_\text{center} = \mathcal{L}_\text{softmax} + \frac{\lambda}{2}\sum_i \lVert z_i - c_{y_i}\rVert_2^2
\]

Claimed benefits (per source):
- mitigates the *expansion issue* by providing explicit class centers
- mitigates the *sampling issue* by reducing reliance on hard mining

Limitations noted conceptually in the source’s progression:
- “single center per class” can struggle with high intra-class variance and noisy labels, motivating multi-center variants later (see Sub-Center ArcFace).

> Related: [[loss_functions]], [[classification]], [[representation_learning]]

---

## Angular-margin losses (SphereFace, CosFace, ArcFace) (new)

A major trend in supervised deep metric learning (especially face recognition and instance retrieval benchmarks) is learning embeddings on a **hypersphere**, using **angular margins** to increase inter-class separation and reduce intra-class variance.

Common setup for these losses:
- normalize classifier weights: \(\|W_j\|=1\)
- normalize embeddings/features: \(\|z\|=1\)
- often set bias \(b=0\)
- use a **scale** parameter \(s\) to keep softmax gradients well-conditioned

### SphereFace (multiplicative angular margin)

- Introduces a multiplicative angular margin \(\mu\) by replacing \(\cos(\theta)\) with \(\cos(\mu\theta)\).
- New source notes optimization complications due to cosine non-monotonicity and dependence of the effective margin on \(\theta\), motivating later variants.

### CosFace (additive cosine margin)

Adds an additive margin \(m\) in cosine space:

\[
\mathcal{L}_\text{CosFace} = -\frac{1}{N}\sum_i \log \frac{\exp(s(\cos\theta_{y_i}-m))}{\exp(s(\cos\theta_{y_i}-m)) + \sum_{j\neq y_i}\exp(s\cos\theta_j)}
\]

Notes from the source:
- choosing \(s\) and \(m\) is important; \(s\) should not be too small (can’t reach confident probabilities) or too large (won’t penalize mistakes).

### ArcFace (additive angular margin)

Defines the margin in angle space by using \(\cos(\theta + m)\):

\[
\mathcal{L}_\text{ArcFace} = -\frac{1}{N}\sum_i \log \frac{\exp(s\cos(\theta_{y_i}+m))}{\exp(s\cos(\theta_{y_i}+m)) + \sum_{j\neq y_i}\exp(s\cos\theta_j)}
\]

Source claim:
- ArcFace is “very similar” to CosFace, but tends to be **slightly better** across benchmarks in reported results.

> Related: [[metric_learning]], [[loss_functions]]

---

## Handling noise, intra-class variance, and imbalance in metric learning (new)

### Sub-Center ArcFace (multiple centers per class)

Motivation:
- A single center per class can be too restrictive when intra-class variance is high or labels are noisy.

Idea:
- each class has \(K\) sub-centers \(\{W_{j,1}\dots W_{j,K}\}\)
- use the closest sub-center for computing the effective angle

Benefit claimed:
- dominant clean modes cluster to main centers; noisy/hard samples can be absorbed by other centers.

### ArcFace with dynamic margin (class-dependent margins)

Motivation:
- extreme class imbalance (long tail) can cause poor convergence for rare classes.

Proposed rule (per source):
\[
m_i = a\cdot n_i^{-\lambda} + b
\]
- rarer classes (smaller \(n_i\)) get larger margins.

> Related: [[class_imbalance]], [[long_tail_distributions]]

---

## Practical “what works” notes from metric learning case studies (new)

The new source summarizes empirical practices from Kaggle-style large-scale retrieval/recognition tasks (images and also text):

- **Shift in popularity (reported):**
  - older competitions: Triplet Loss and variants were common
  - later competitions (e.g., Google Landmarks 2020): ArcFace/CosFace variants used extremely widely among top solutions
- **ArcFace/CosFace can be used beyond images:**
  - reported to work for **text embeddings** as well (e.g., product matching with both image and text towers)
- **Post-processing often matters in retrieval:**
  - metric learning alone may not be sufficient; solutions often use query expansion and verification/matching steps
- **Hyperparameters differ by modality:**
  - optimal \((s,m)\) may differ for image vs text models.

Operationally, this fits the broader retrieval theme:
- embedding learning is only part of the system; indexing, ANN, and downstream heuristics matter too.

> Related: [[information_retrieval]], [[embedding_models]], [[vector_search]]

---

## Practical takeaways (fundamentals emphasized by the source)

- Deep learning in retrieval/ranking is often about **representation learning under latency constraints**.
- Two-tower models are foundational because they support:
  - scalable indexing,
  - decoupled inference,
  - fast similarity computation.
- Modern extensions (DAT, IntTower, late-interaction models like ColBERT) attempt to mitigate the classic **lack of interaction** while retaining efficiency.
- Deep metric learning provides a complementary lens:
  - many embedding models can be trained with **pair/triplet** losses or **angular-margin softmax** losses to produce more discriminative spaces.

---

## Contradictions / tensions to note (updated)

### 1) “InfoNCE-style contrastive loss” vs “move away from contrastive approaches”
- **Existing page:** IntTower’s CIR module uses an **InfoNCE-style contrastive loss** as a helpful regularizer.
- **New source:** argues that in *supervised deep metric learning*, the field “moved away” from direct \(l_2\)-contrastive / triplet-style objectives due to **sampling** and **expansion** issues, favoring **angular-margin** softmax-style objectives (ArcFace/CosFace).

This is not a strict contradiction, but it is a **contextual tension**:
- In retrieval/pre-ranking, InfoNCE-style objectives can be effective and scalable with