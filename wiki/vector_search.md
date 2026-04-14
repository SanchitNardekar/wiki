---
slug: vector_search
sources:
- hav4ik.github.io
- blog.reachsumit.com
- relevance_filtering_for_embedding_based_retrieval.pdf
tags: []
title: Vector Search and ANN
updated: '2026-04-14'
---

# Vector Search and ANN

Vector search (also called embedding-based retrieval) is a family of retrieval techniques where both queries and documents/items are represented as dense numeric vectors (“embeddings”), and retrieval is performed by finding the nearest vectors according to a similarity function (commonly cosine similarity or Euclidean distance). In modern search stacks, vector search is typically used for *top‑k retrieval* (a.k.a. “matching” or “Level‑0 ranking”), and is often combined with keyword-based retrieval.

This page focuses on **vector search** and **Approximate Nearest Neighbor (ANN)** methods that make nearest-neighbor retrieval feasible at large scale, and how they fit into a broader search system and ranking pipeline (see also [[learning_to_rank]]).

**New in this update:** additional context on **relevance filtering / score calibration for dense retrieval** (CIKM’24 “Cosine Adapter”), motivated by the fact that ANN-based dense retrieval often has **no natural cutoff** and raw cosine scores are frequently **not comparable across queries**.

---

## Where vector search fits in a modern search engine

A common high-level architecture for search systems is:

1. **Indexing (offline / continuous)**: extract signals and build indexes.
2. **Top‑k retrieval (online)**: quickly fetch a candidate set of potentially relevant documents.
3. **Ranking / re-ranking (online)**: sort candidates by relevance and/or personalization, often with ML-based Learning-to-Rank.

Vector search primarily powers step (2), but also produces features that can be used in ranking/re-ranking stages (step (3)).

### Cascade / multi-stage ranking systems (retrieval → pre-ranking → ranking)

The new source expands the above pipeline into a **cascade (multi-stage) ranking system**, motivated by huge candidate sets (tens of millions) and strict latency constraints:

- **Recall / Retrieval**: very fast, recall-focused candidate generation (often lexical + vector retrieval).
- **Pre-ranking**: an intermediate stage that filters or scores a larger candidate set than the final ranker can handle; commonly uses *fast neural models* such as two-tower architectures.
- **Ranking / Re-ranking**: more expensive models (often deep neural networks and/or LTR) applied to a smaller candidate set.

Latency motivation from the source:
- Even ~**100ms** response-time increases can measurably degrade user experience and revenue, so systems adopt multiple stages rather than one complex ranker.

Cross-references:
- Ranking and re-ranking with ML: [[learning_to_rank]]

### Index types commonly used together

Modern web-scale and recommender-style search engines may maintain multiple indexes, each optimized for different retrieval paradigms:

- **Inverted index (posting lists)**
  - Maps terms → documents.
  - Enables classic lexical retrieval and scoring such as TF‑IDF and BM25.
  - Useful for exact term matching and entity matching.

- **Vector index**
  - Stores dense embeddings for documents (and sometimes pre-computed query embeddings, depending on the system).
  - Enables nearest-neighbor retrieval by vector similarity (cosine or Euclidean).

- **Feature index**
  - Stores a large collection of engineered signals and/or compressed neural embeddings.
  - Used heavily in later-stage ranking (re-ranking).

This “multi-index” view aligns with modern search stacks where retrieval is **hybrid**: lexical + embedding-based retrieval, sometimes augmented by query expansion via knowledge graphs.

Cross-references:
- Ranking and re-ranking with ML: [[learning_to_rank]]
- Classical lexical retrieval signals (e.g., BM25): [[bm25]] (if present)

---

## Embeddings for vector search

Vector search depends on embeddings learned by neural models, commonly via **contrastive learning** approaches.

Examples mentioned in the source material:

- **Visual search**: embeddings such as **SigLIP**-style representations.
- **Text search**: **BERT-like** embeddings.

Typical workflow:

- Compute document embeddings during indexing and store them in the **vector index**.
- Compute a query embedding at request time.
- Retrieve the **k nearest document vectors** by similarity.

### Two-tower (dual-encoder / bi-encoder) models as a common embedding generator

The new source adds detail on a dominant industrial approach for producing embeddings used in retrieval and pre-ranking: the **two-tower model** (also known as a **dual encoder** / **bi-encoder**).

Key properties (from the new source):

- **Two towers** encode the two sides independently (e.g., **query/user** vs **document/item/ad**) into latent vectors.
- A **similarity score** is computed via **inner product** between the two vectors (often used as a pre-ranking score; can also be used directly for retrieval).
- The model is designed for **inference efficiency**:
  - the two towers compute embeddings in parallel,
  - they interact mainly at the output layer (**late interaction** in the two-tower sense).

Operational implication for vector search systems:
- It is common to **freeze item/document embeddings after training** and store them in an indexing service (i.e., the vector index), enabling fast retrieval/scoring at serving time.

Cross-references:
- Ranking and multi-stage pipelines: [[learning_to_rank]]

---

## Similarity functions used for nearest neighbors

Common choices:

- **Cosine similarity**
- **Euclidean distance**

Implementation note: many ANN systems support multiple metrics or implement cosine similarity via normalized vectors and inner product.

### Inner product / dot product in two-tower systems

The new source specifically highlights **inner product** as the standard similarity function used by two-tower models (e.g., user-query vector ⋅ item-doc vector) to produce a retrieval/pre-ranking score.

**Compatibility note (no contradiction):**
- Inner product is consistent with existing content: cosine similarity can be implemented as inner product on **L2-normalized** vectors, and many ANN systems expose **maximum inner product search (MIPS)** as a supported mode.

---

## Why Approximate Nearest Neighbor (ANN) is used

Exact nearest neighbor search over large corpora can be too slow and memory-intensive. The source text highlights a key practical point for **large-scale** search engines:

- **Metric trees** (e.g., k‑d trees) are described as *not used* in large-scale search engines due to:
  - slow behavior at scale despite the nominal $O(\log n)$ complexity, and
  - large memory consumption.

Instead, large systems use **Approximate Nearest Neighbor (ANN)** methods to achieve **close to $O(1)$** retrieval time in practice.

> Note: The “close to $O(1)$” claim reflects an engineering perspective and typical behavior of hashed/quantized ANN lookups, not a universal theoretical guarantee for all ANN methods and all distributions.

---

## ANN families and techniques (from the new source)

The source explicitly mentions ANN approaches such as:

- **Hashing-based ANN**
  - e.g., “LHS” (likely intended as LSH / Locality-Sensitive Hashing; the source text uses “LHS”)
  - **PCA hashing**

These approaches trade exactness for speed by mapping vectors into codes/buckets so that candidate neighbors can be found without scanning the entire dataset.

### Contradiction / ambiguity note
- The source states “ANN search (like **LHS** or **PCA hashing**)”. In common terminology, **LSH** (Locality-Sensitive Hashing) is the standard acronym; “LHS” is ambiguous and may be a typo or refer to a different method. This page preserves the source phrasing but flags the ambiguity.

---

## Deep Metric Learning (DML) as a foundation for embedding quality (new)

Many embedding models used for vector search can be viewed through the lens of **metric learning**: learning an embedding function $f_\theta(x)\in\mathbb{R}^n$ such that distances/similarities in embedding space correspond to semantic similarity.

From the new source (a DML survey), the supervised metric learning problem is described as choosing:

- an embedding model $f_\theta(\cdot)$ (feature extractor), and
- a distance function $\mathcal{D}(\cdot,\cdot)$ (often fixed, e.g., $L_2$),

so that:
- $\mathcal{D}(f_\theta(x_1), f_\theta(x_2))$ is **small** when labels match,
- and **large** when labels differ.

### Why this matters for vector search

- Vector search quality (recall/precision at top‑k) depends heavily on **embedding geometry**:
  - how tightly “similar” items cluster,
  - how well-separated “dissimilar” items are,
  - how stable similarity scores are across classes/domains (e.g., long-tail, noisy labels, out-of-distribution queries).

- DML objectives can implicitly choose which similarity metric is most appropriate at serving time:
  - Many classic DML losses operate explicitly in **Euclidean ($L_2$) distance** space.
  - Many modern “angular margin” losses operate on **cosine similarity** after normalization, aligning closely with cosine/dot-product-based ANN retrieval.

**Mild tension / nuance with existing content (not a direct contradiction):**
- The existing page emphasizes cosine/Euclidean as retrieval metrics and notes dot-product/MIPS for two-tower retrieval.
- The new DML source emphasizes that many “direct” metric learning methods are typically defined in **$L_2$** space, while later “angular” methods explicitly normalize features and optimize **cosine/angular separation**.  
  Reconciliation: both families ultimately produce embeddings that can be used in ANN; you should ensure the ANN metric (cosine vs $L_2$ vs MIPS) matches the training objective and any normalization steps.

Cross-references:
- Training/ranking pipeline context: [[learning_to_rank]]

---

## Contrastive / “direct” metric learning losses (new)

The DML survey describes early/common supervised metric learning approaches as “contrastive” (sometimes called “direct”), because they directly pull positives together and push negatives apart, typically under $L_2$.

### Contrastive loss (pairwise)

Given two samples $(x_1,y_1)$, $(x_2,y_2)$, distance $\mathcal{D}$ (often $L_2$), and margin $\alpha$:

- Pull together same-label pairs via $\mathcal{D}^2$
- Push apart different-label pairs by enforcing a margin

Key practical note from the source:
- The margin prevents a degenerate solution where the model maps everything to the same point.

### Triplet loss (anchor/positive/negative)

Triplet loss uses:
- anchor $x_a$,
- positive $x_p$ (same label),
- negative $x_n$ (different label),

and enforces:
- anchor-positive closer than anchor-negative by a margin $\alpha$.

#### Negative sample mining (important operational detail)

The source highlights that triplet loss performance depends heavily on **negative mining**: selecting “useful” negatives that violate (or nearly violate) the desired margin constraint.

**Engineering implication for retrieval embeddings:**
- The embedding quality you get for vector search is often bounded by how well you can:
  - find hard negatives,
  - scale mining in distributed training,
  - avoid training collapse or slow convergence when most negatives become “easy”.

### Limitations that motivate newer objectives (from the source)

The DML survey argues that directly optimizing pair/triplet distances has two main issues:

- **Expansion issue:** hard to ensure all same-label samples collapse into a single coherent region *globally* (many objectives enforce local/batch structure).
- **Sampling issue:** reliance on mining is inconvenient locally and can be problematic at distributed scale.

---

## Moving from Euclidean “direct” losses to angular/cosine margin losses (new)

The DML survey describes a shift (notably after ~2017 in face recognition) toward objectives that improve class separability by operating on **angles / cosine similarity** with normalized features and weights.

This is particularly relevant to vector search because:

- cosine similarity is a standard retrieval metric,
- normalized embeddings make dot product equivalent to cosine,
- angular margins can yield more discriminative neighborhoods for nearest-neighbor retrieval.

### Center Loss (regularizing softmax features)

The source describes **Center Loss** as augmenting standard softmax classification loss with an $L_2$ penalty that pulls embeddings toward a per-class center $c_{y_i}$.

Claimed benefits in the source:
- Helps address the expansion issue by providing explicit centers.
- Reduces reliance on hard mining (mitigates the sampling issue).

### SphereFace → CosFace → ArcFace (large-margin angular/cosine losses)

The source presents a progression of “angular margin” methods:

- **SphereFace (2017):** introduces multiplicative angular margin but has optimization complications (non-monotonicity of cosine; margin depends on $\theta$; requires piecewise tricks).
- **CosFace (2018):** introduces an **additive cosine margin** with normalized features and weights; uses scale $s$ and margin $m$.
- **ArcFace (2019):** introduces an **additive angular margin** in angle space; similar setup (normalized features/weights, scale $s$, margin $m$); reported to slightly outperform CosFace in many benchmarks.

#### Hyperparameters $s$ (scale) and $m$ (margin)

The source emphasizes that for CosFace/ArcFace:
- choosing $s$ and $m$ is crucial,
- too small/large $s$ can harm training dynamics and calibration of probabilities,
- increasing $m$ effectively enforces stricter separation (shifts probability curves).

It also cites **AdaCos** (2019) as proposing a heuristic fixed scaling:
- $\tilde{s} \approx \sqrt{2}\log(C-1)$ where $C$ is the number of classes,
and notes (anecdotally in the blog) that “Adaptive AdaCos” is not commonly seen deployed successfully.

**Connection to vector search serving:**
- If embeddings are L2-normalized (as in CosFace/ArcFace setups), then:
  - cosine similarity and dot product become equivalent,
  - ANN configured for cosine (or inner product on normalized vectors) naturally matches training.

### Handling intra-class variance and imbalance: Sub-center ArcFace + Dynamic Margin

The source introduces two extensions:

- **Sub-center ArcFace (2020):**
  - multiple centers per class, choosing the closest sub-center.
  - motivation: a single center is a poor fit when intra-class variance is high and labels are noisy.

- **ArcFace with Dynamic Margin (2020):**
  - uses per-class margin $m_i = a\cdot n_i^{-\lambda} + b$ based on class frequency $n_i$,
  - motivation: extreme class imbalance; smaller classes may need larger margins.

**Implication for retrieval embeddings:**
- These methods aim to produce embeddings with:
  - more robust clusters under noise,
  - better behavior for long-tail/rare classes,
  - potentially improved nearest-neighbor neighborhoods (especially top‑k purity) in imbalanced catalogs.

---

## Hybrid retrieval: vectors + keywords

For web-scale search engines, retrieval often uses a **hybrid** of:

- **Keyword / entity matching** via the inverted index
- **Embedding-based retrieval** via the vector index

This improves recall (semantic matching) while retaining precision/constraints provided by lexical matching. Hybrid retrieval is especially useful when:
- queries are short/ambiguous,
- vocabulary mismatch exists (synonyms, paraphrases),
- users search with natural language.

---

## Relationship to ranking and Learning to Rank (LTR)

Vector search is typically used to fetch a candidate set (top‑k) which is then refined by a ranker. The new source emphasizes:

- **Retrieval** (“Level‑0 ranking” / “matching”) produces top‑k candidates.
- **Ranking** is what “actually makes search engines work” by ordering retrieved documents by relevance (and optionally user preferences).
- Large search engines rely on **Machine Learning / Learning-to-Rank** in later stages.

Cross-reference: [[learning_to_rank]]

### Where two-tower models often sit: pre-ranking

The new source positions two-tower models as a **go-to architecture for pre-ranking** in industrial pipelines (ads/recs/search), because:

- pre-ranking must score **many more candidates** than the final ranker,
- two-tower inference is efficient due to independent embedding computation,
- embedding indexes allow low-latency scoring and retrieval.

**Potential nuance / mild tension with existing phrasing (not a direct contradiction):**
- Existing content frames vector search primarily as step (2) “top‑k retrieval”.
- The new source emphasizes that, in many real systems, there is an explicit **pre-ranking stage** between retrieval and ranking where two-tower models are heavily used.
- Reconciliation: vector search can be used in **retrieval**, and also to support **pre-ranking** (e.g., embedding-based scoring of a retrieved candidate set), depending on the stack.

---

## Dual-encoder architecture variants and retrieval quality implications (from the new source)

The new source adds practical modeling distinctions that matter for embedding-based retrieval quality:

- **Siamese Dual Encoder (SDE)**:
  - two identical encoders, typically with shared parameters.
  - reported (in the cited study) to perform better than fully asymmetric dual encoders in a QA retrieval task.

- **Asymmetric Dual Encoder (ADE)**:
  - two different encoders with different parameters.
  - can suffer if the two sides land in “disjoint” embedding spaces, hurting retrieval quality.

- Reported mitigation:
  - **sharing a projection layer** between encoders (ADE-SPL) can improve ADE performance to be on par with or better than SDE in the referenced study.
  - sharing token embedders (ADE-STE) or freezing token embedders (ADE-FTE) gave only marginal improvements in that study.

Operational tie-back to vector search:
- These architectural choices change how “well-aligned” query and document vectors are, which directly impacts nearest-neighbor retrieval quality under cosine/inner-product similarity.

---

## Interaction-focused neural ranking vs representation-based retrieval (context for vector search)

The new source contrasts two-tower (representation-based) retrieval models with more interaction-heavy rankers:

- **Two-tower / representation-based**: compute independent embeddings, compare at the end (efficient; index-friendly).
- **Interaction matrix models** (e.g., DRMM, KNRM): model term/phrase interactions; typically heavier than pure dual encoders.
- **Cross-encoders** (e.g., BERT cross-encoder): compute query-document interactions jointly; typically the most accurate but expensive.
- **Late-interaction hybrids** (e.g., ColBERT): preserve “query-document decoupling” while enabling richer interactions than pure two-tower.

**How this fits with ANN/vector indexes:**
- The more a model preserves **decoupled document representations** (document vectors can be precomputed/frozen), the more naturally it fits **vector indexing + ANN** at serving time.
- Interaction-heavy models are more likely to be used in **later ranking stages** rather than first-stage ANN retrieval.

Cross-references:
- Multi-stage ranking / re-ranking: [[learning_to_rank]]

---

## Relevance filtering for embedding-based retrieval (Cosine