---
slug: data_pipelines
sources:
- hav4ik.github.io
tags: []
title: Data Pipelines
updated: '2026-04-14'
---

# Data Pipelines

Data pipelines are end-to-end, repeatable workflows that move data (or other artifacts such as model outputs) through a sequence of stages—typically including ingestion, filtering, transformation, validation, storage, and evaluation—so that downstream systems can reliably consume the results.

This page focuses on **practical pipeline design**, with an emphasis on **ML/LLM training and evaluation pipelines** where “data” often means *prompts, reasoning traces, rollouts, rewards, and metrics* rather than just rows in tables.

## Why data pipelines matter (especially for LLM/RL)

Well-designed data pipelines make it possible to:

- **Scale experiments** without losing reproducibility.
- **Control quality** (e.g., remove duplicates, filter low-signal examples).
- **Separate concerns** (e.g., skill acquisition vs behavior shaping).
- **Track tradeoffs** (accuracy vs length, compute vs dataset size).
- **Avoid contamination** by choosing time-separated benchmarks and known data provenance.

In LLM reasoning work, pipelines often become the *main product*: the training and evaluation loop can dominate outcomes more than any single modeling trick.

## Core stages of a modern (LLM) data pipeline

### 1) Source selection & ingestion
Choose raw sources that match the target distribution and difficulty. In the referenced math reasoning work, the initial pool was drawn from **NuminaMath-1.5** (math word problems across Algebra/Geometry/Number Theory/Combinatorics, sourced from Olympiads, AoPS, AMC/AIME, etc.).

Key practices:
- Prefer sources with **known provenance** and **difficulty aligned** to the target task.
- Maintain a catalog of sources, versions, and filtering rules.

### 2) Joining / enrichment with additional signals
Pipelines often join raw problems with:
- **Existing solution traces** (e.g., “correct reasoning traces”)
- Metadata (topic, difficulty proxies, length, etc.)

Example pattern:
- Join filtered problems with correct reasoning traces from **OpenR1-Math-220k**, then downselect.

### 3) Filtering (quality, difficulty, length)
Filtering is usually multi-pass:

- **Length filtering**
  - Example: collecting solution traces under **16K tokens** because most correct outputs were under ~6K and 16K balanced accuracy vs compute.
- **Difficulty filtering via sampling**
  - Example: sample multiple solutions per problem (e.g., 8 rollouts) with a given model and remove “too easy” items by keeping problems with ≤7 correct solutions.
- **Duplicate removal**
  - Example: adding a subset of **Light-R1** stage-2 data, then removing duplicates against the existing dataset.

Pipeline takeaway: difficulty filtering can be *model-dependent* and is often implemented as a “generate → score → filter” loop.

### 4) Dataset partitioning by purpose
A common, effective pattern is to split data by intended training stage:

- **SFT set** (broader, high-quality)
- **RL/GRPO set** (harder subset to provide learning signal)

Example:
- Final SFT dataset: **10K+ samples** (e.g., ~8K from NuminaMath filtering + ~2K from Light-R1 subset)
- Harder half used for [[reinforcement_learning]] stage.

This supports **separation of concerns**:
- SFT = skill acquisition
- RL = behavior steering (e.g., shorter chains, different style)

### 5) Training pipeline orchestration (SFT → RL)
A pipeline is not just data movement; it includes compute stages and artifacts:

- **Stage 1: SFT**
  - Example: training on 8×H100 for multiple epochs at long context (16K).
  - Note: longer fine-tuning can improve accuracy but can also create unnecessarily long chains-of-thought (length inflation).
- **Stage 2: RL (GRPO)**
  - Example: GRPO runs at 8K then 16K context for 7B; exploration of context length strategies for 14B.
  - Integrations:
    - DAPO-style clipping and online sample filtering
    - length penalties sometimes help at smaller scales but can harm larger models

Cross-reference: [[training]], [[fine_tuning]], [[evaluation]].

## Data pipeline meets RL pipeline: GRPO-specific operational issues

The source text highlights that in long-context RL for reasoning models, the “pipeline” must address algorithmic biases and scaling bottlenecks.

### Length bias (“length hacking”) in GRPO
Observed issue:
- During training, solutions get longer over time, especially for SFT-distilled reasoning models.

Claim from new source:
- The length bias is largely an **implementation issue**:
  - If losses are normalized per-sample (averaged within each sample), tokens in longer sequences may contribute less to the total loss.
  - Fix: remove per-sample normalization and compute loss uniformly over all tokens.

Related methods mentioned:
- **DAPO** and **Dr. GRPO**: remove/adjust normalization terms that induce implicit length bias.

### Difficulty bias from reward scaling
In original GRPO, advantage normalization includes a scaling term based on reward standard deviation:
- This can bias training toward very easy/hard prompts (rewards near all-1 or all-0).
- Dr. GRPO addresses by removing reward scaling.

### KL regularization uncertainty
Discussion points:
- GRPO uses **forward KL** \(D_{KL}(\pi_\theta \| \pi_{ref})\) rather than PPO’s reverse KL.
- Experiments reported negligible differences in some settings.
- DAPO argues KL can be removed entirely for long-CoT because the model diverges enough that regularization may not help.

**Potential contradiction to note:** In many RLHF recipes (common prior art), KL is considered critical to prevent drift; the new source suggests KL can be less important (or removable) for long-CoT reasoning. This is not necessarily a direct contradiction (different regimes), but it is a notable *tension* in guidance.

Cross-reference: [[rlhf]] (if present), [[kl_divergence]] (if present), [[reinforcement_learning]].

## Online curriculum & stabilization in the pipeline

A pipeline for RL is not stable if batches frequently contain prompts with zero learning signal.

### Failure mode: all rollouts get reward = 0
Observed behavior:
- When hard prompts yield all-zero rewards, training can “spiral” into garbage outputs (including language mixing).
- Fix: include more easy/medium items to stabilize signal.

### Online filtering as curriculum (DAPO)
DAPO-style pipeline behavior:
- Oversample prompts each step
- Filter out too-easy and too-hard prompts
- Train on the selected subset

Practical takeaway: a “data pipeline” can be **adaptive** at runtime, not only static preprocessing.

## Evaluation pipelines (metrics, sampling, contamination control)

Evaluation is a first-class pipeline stage, not an afterthought.

### Benchmark selection & contamination control
Example approach:
- Use a benchmark released **after** base model training as “uncontaminated” (e.g., AIME 2025 after DeepSeek R1 training).
- Ensure main dataset was collected before the benchmark release.

### Sampling & aggregation pipeline
Reported evaluation pipeline settings included:
- Generate a pool of rollouts per question (e.g., 64 traces)
- Compute:
  - Pass@1 and average length over the full pool
  - Majority-vote metrics (Maj@K) by subsampling K traces repeatedly and averaging to reduce noise

Important insight:
- **More token budget can reduce Maj@32** because models may self-doubt or drift with extra “thinking time”.

**Explicit tension to track:** Many practitioners assume “more tokens / more thinking time ⇒ better accuracy.” The source reports a counter-effect where longer budgets can degrade majority-vote accuracy. This does not contradict general scaling laws outright, but it contradicts a common operational assumption in evaluation pipeline design.

Cross-reference: [[benchmarking]], [[model_evaluation]] (if present).

## Model artifact pipeline: merging & deployment artifacts

Beyond training checkpoints, pipelines often include:
- Quantization artifacts (e.g., AWQ)
- Model merging

Example:
- Merge SFT and GRPO checkpoints using **MergeKit** with **TIES** merging.
- Observations:
  - For 7B, merging improved both accuracy and “token economy.”
  - For 14B, merging became more of a compromise.

Operational note:
- Quantization can erase part of training gains, so deployment pipelines should include quantization-aware validation.

Cross-reference: [[model_merging]], [[quantization]].

## Engineering considerations: pipeline performance & scaling bottlenecks

Long-context RL turns pipeline orchestration into systems engineering.

### Framework choice as pipeline infrastructure
Capabilities required for long-context GRPO pipelines include:
- Sequence packing, long-sequence parallelism (e.g., “Ulysses”)
- FSDP and CPU offloading
- Model colocation (actor/ref/rollout engine) for memory savings
- vLLM rollouts and efficient batching

The source contrasts veRL-style systems vs earlier limitations in some trainers, and notes that tooling evolved quickly.

### Throughput bottleneck: rollouts dominate
In long-context GRPO:
- “Rollout” time can exceed optimization time.
- Idle time appears due to waiting for the longest sequence in a batch.

A mitigation described:
- Reuse rollouts across multiple optimization steps (“data echoing”-like), with limited lag in policy updates.

Cross-reference: [[mlops]] and [[distributed_training]] (if present).

### LoRA integration complexity (pipeline risk)
While [[lora]] is often considered simple for SFT, integrating it into an RL pipeline with:
- FSDP sharding
- reference models
- vLLM offloading
can be operationally difficult (dtype mismatches, VRAM spikes during merges, startup overlap).

Pipeline takeaway: adapter-based fine-tuning can reduce VRAM but increase *systems complexity*.

## Practical pipeline patterns (checklist)

- **Keep an audit trail**: source dataset versions, filters, dedupe hashes, sampling seeds.
- **Filter by measured difficulty**: generate multiple candidate solutions and filter based on correctness rate.
- **Control length distribution**:
  - cap trace length (e.g., 16K)
  - detect and mitigate GRPO length bias in the loss implementation
- **Stabilize RL signal**:
  - avoid batches with all-zero rewards
  - use online filtering/curriculum
- **Evaluate with multiple metrics**:
  - Pass@1, Maj@K, average length, token budget sweeps
  - beware “more thinking time” hurting majority-vote accuracy
- **Validate after each artifact transformation**:
  - merging, quantization, prompt/stopping-condition changes

## Related pages
- [[reinforcement_learning]]
- [[fine_tuning]]
- [[evaluation]]
- [[model_merging]]
- [[quantization]]
- [[mlops]]
- [[distributed_training]]
- [[lora]]