---
slug: model_serving
sources:
- hav4ik.github.io
tags: []
title: Model Serving and Inference
updated: '2026-04-14'
---

# Model Serving and Inference

This page covers practical considerations for serving and running inference for modern LLMs and reasoning models, with emphasis on *long-context* and *test-time scaling* workloads (e.g., sampling many reasoning traces and aggregating via majority vote). It also highlights how training choices (SFT/RL methods like GRPO) materially affect inference-time behavior such as output length, latency, stability, and cost.

Related pages (cross-reference as you build out the wiki):
- [[reinforcement_learning]] (GRPO, PPO, REINFORCE++)
- [[rlhf]] (KL regularization, policy optimization variants)
- [[fine_tuning]] (SFT vs RL stages, LoRA vs full fine-tuning)
- [[lora]] (adapter serving/merging implications)
- [[vllm]] (rollout/inference engine used in long-context RL and scalable inference)
- [[fsdp]] (training-time sharding/offload that impacts deployment artifacts)
- [[evaluation]] (Pass@1, Maj@K, token budgets, sampling methodology)
- [[quantization]] (AWQ and accuracy regressions)
- [[model_merging]] (MergeKit/TIES and deployment tradeoffs)
- [[prompting]] (prompt-induced metric shifts, stopping conditions)

---

## Scope: what “serving and inference” means for reasoning models

For reasoning-focused LLMs (e.g., DeepSeek-R1 distilled models and derivatives), inference is often not “single-shot generation”. Instead, common inference patterns include:

- **Multi-sample decoding**: generate many independent solution traces per prompt.
- **Aggregation**: compute **Maj@K** (majority voting) from K sampled traces.
- **Budgeted thinking**: impose a maximum generation length (`max_len`) or total token budget.
- **Stop conditions**: choose to stop at EOS vs custom stop sequences to reduce wasted tokens.

This inference style changes serving requirements:
- Throughput depends on *tokens generated across all rollouts*, not just requests/sec.
- Tail latency is dominated by the *longest* generation in a batch/group.
- Memory pressure increases due to long sequences and large KV caches.

---

## Key inference metrics and how they interact

### Pass@1 vs majority voting (Maj@K)
A key practical inference lesson from recent long-context math-reasoning work is:

- **Better Pass@1 does not imply better Maj@K** (e.g., Maj@32).
- Models can produce more diverse (or more self-contradictory) traces when given larger token budgets, which can **reduce majority-vote accuracy** even when they “think longer”.

This is operationally important: if your product relies on self-consistency / majority voting, you should optimize serving for **Maj@K under a target token budget**, not only Pass@1.

Cross-reference: [[evaluation]]

### Token budget effects (counterintuitive behavior)
Empirical observation on math reasoning benchmarks (AIME’25) indicates:

- Increasing token budget beyond a point (e.g., from ~16K to ~32K) can **slightly worsen Maj@32**.
- A hypothesized mechanism is **self-doubt**: with more “thinking time,” the model may revise correct reasoning into incorrect conclusions.

Implication for serving:
- “More tokens” is not always “better answers.”
- Prefer **budget sweeps** to identify the peak accuracy-per-token operating point.

---

## Sampling settings commonly used for reasoning inference

A representative inference/evaluation configuration used for long-form reasoning traces:

- **Precision**: BF16
- **Sampling**: `temperature=0.75`, `top_p=0.95`
- **Rollouts collected per question**: 64
- **Max generation length**: up to 32768 tokens (for trace collection), with smaller limits (e.g., 12800 or 16384) used for some aggregated metrics.
- **Maj@K computation**: sample K traces from the pool; repeat aggregation multiple times (e.g., 16) and average to reduce noise.

Operational notes:
- This setup is expensive: inference cost is dominated by generating many long traces.
- When comparing models, keep prompts and decoding parameters fixed; otherwise you can get misleading gains.

Cross-reference: [[evaluation]], [[prompting]]

---

## Stop conditions and early stopping

One evaluation approach uses a custom stop sequence (e.g., `stop=['']`) for certain fine-tuned models because:

- They were trained with more restricted generation lengths and often output a correct answer *before completing* a long chain-of-thought.
- Applying the same stop condition to baseline DeepSeek-R1 models led to noticeable accuracy drops; thus those baselines were evaluated with **default EOS stopping**.

Serving implication:
- **Stop strategy is model-dependent**; a stop sequence that improves efficiency for one model may harm another.
- If you deploy a mixture of models, implement stop conditions as **per-model configuration**.

Cross-reference: [[prompting]]

---

## Length behavior at inference time (and why training matters)

### Length growth from SFT
SFT on reasoning traces can cause solutions to become progressively longer with more training, especially for models distilled from very large reasoning models (e.g., DeepSeek-R1 671B).

Serving implications:
- Expect rising **average tokens per response** over training iterations unless you explicitly control it.
- Rising length increases:
  - Latency
  - Cost
  - Tail risk (timeouts / truncation)
  - Majority-vote economics (fewer rollouts within a budget)

Cross-reference: [[fine_tuning]]

### GRPO “length hacking” and mitigation
In GRPO-style training, “length hacking” can occur where:
- Longer incorrect solutions may be penalized less (token-wise) than short incorrect ones, depending on how loss is normalized.

Two mitigation approaches cited:
- **Remove per-sample loss normalization** and compute loss uniformly across all tokens (a fix framed as largely an implementation issue).
- Apply algorithmic updates from works like **DAPO** and **Dr. GRPO**, which target:
  - implicit length bias
  - advantage scaling / difficulty bias
  - (sometimes) removing or modifying the KL term

Serving implications:
- If training reduces length without harming accuracy, you can:
  - increase number of rollouts K under the same budget
  - reduce per-request cost at the same quality
- But length penalties can also **hurt accuracy**, especially at larger model sizes (see “Contradictions / tradeoffs” below).

Cross-reference: [[reinforcement_learning]], [[rlhf]]

---

## Contradictions / tradeoffs to note explicitly

New source information contains several tensions that affect how you should design inference serving:

- **Length penalty helps at 7B but hurts at 14B**
  - Observed: length penalty “worked well” for a 7B GRPO model, but “severely hurts accuracy” for 14B, leading to removal at 14B (keeping only outcome reward).
  - Serving consequence: you cannot assume the same decoding/length-control policy (or training-side length shaping) scales uniformly with model size.

- **Longer inference budgets can reduce Maj@32**
  - Contrary to the common intuition that more compute improves aggregation quality, the reported results show a slight degradation in Maj@32 at very long budgets (e.g., 32K vs 16K).
  - Serving consequence: hard-cap budgets and empirically tune them rather than defaulting to “as long as possible.”

- **Prompting differences can change headline Pass@1**
  - Reported inability to reproduce third-party Pass@1 for a 32B model, likely due to a prompt that hints the answer is an integer.
  - Serving consequence: “accuracy” is not model-only; it’s *model × prompt × decoding × stop condition*.

Cross-reference: [[prompting]], [[evaluation]]

---

## Test-time scaling economy (accuracy vs tokens)

A useful way to think about inference serving for reasoning is as an **economy curve**:

- x-axis: total token budget (or average tokens generated)
- y-axis: accuracy (Pass@1, Maj@32, etc.)

Reported findings on AIME’25 suggest:
- Some fine-tuned/GRPO models can reach the baseline model’s peak Maj@32 with **~33% fewer tokens** (i.e., better “test-time scaling economy”).
- Models trained on longer traces may outperform at *very* long budgets (24K–32K), while shorter-trace-trained models can be more efficient and better up to ~16K.

Serving implications:
- Pick operating points aligned with product constraints:
  - latency SLOs
  - token cost budgets
  - whether you run K-sample voting
- Consider dynamic policies:
  - short budget for most requests
  - longer budget only when confidence is low (requires a confidence signal)

Cross-reference: [[evaluation]]

---

## Inference stack and engineering considerations (long-context)

### vLLM rollouts and batching effects
Long-context reasoning inference often uses an optimized engine (e.g., [[vllm]]) for generation/rollouts. In long sequences:

- The **rollout phase** can dominate wall-clock time.
- **Idle GPU gaps** occur while waiting for the longest sequence in a batch/group to finish.
- Effective batching must account for the heavy-tailed distribution of completion lengths.

Serving recommendations:
- Use **sequence packing** where possible (especially in training/rollout collection; production applicability depends on workload).
- Consider grouping requests by:
  - expected length
  - task type
  - sampling mode (single vs multi-rollout)

Cross-reference: [[vllm]]

### Collocation / “hybrid engine” ideas (training-to-serving relevance)
While discussed in the context of long-context RL, the systems concepts generalize:

- Collocating multiple model roles in one memory space (e.g., actor/ref/engine) is used to manage VRAM pressure.
- For serving, analogous constraints show up when you host:
  - multiple adapters (LoRA)
  - multiple quantized variants
  - or multiple replicas for throughput

Cross-reference: [[fsdp]], [[lora]], [[vllm]]

---

## LoRA, full fine-tuning, and deployment implications

### LoRA convergence vs serving complexity
LoRA can converge faster and be more VRAM-efficient in some GRPO settings, but it introduces real deployment complexity, especially when combined with:

- FSDP sharding and DTensor/dtype pitfalls
- merging adapter weights for inference engines
- VRAM spikes during merge + engine startup (noted with vLLM offloading)

Serving implications:
- Decide early whether you will deploy:
  - **base model + adapters** (hot-swappable but more moving parts), or
  - **merged weights** (simpler serving artifact; may require offline merge pipeline)

Cross-reference: [[lora]], [[fsdp]], [[vllm]]

---

## Model merging and serving artifacts

### Merging SFT and GRPO checkpoints (TIES via MergeKit)
A practical approach for producing a deployable checkpoint:
- Merge multiple checkpoints (e.g., SFT + GRPO) using **TIES**.
- In one reported setup, hyperparameters were not tuned (weights=1, density=1).

Observed behavior:
- For **7B**, merging improved both accuracy and “token economy,” surpassing SFT and GRPO checkpoints.
- For **14B**, merging was more of a tradeoff/compromise (not a strict win).

Serving implications:
- Merging can be a way to:
  - simplify deployment (single checkpoint)
  - recover a Pareto point (accuracy vs length)
- But merging effects are scale-dependent; validate per model size.

Cross-reference: [[model_merging]]

---

## Quantization considerations (deployment risk)

A notable deployment pitfall:
- Quantization (e.g., AWQ variants uploaded for serving) can **erase a sizeable part of training gains**.

Serving implications:
- Always evaluate **the exact artifact** you deploy (quantized weights + engine + decoding settings).
- Consider having:
  - a BF16 “gold” model for periodic regression tests
  - an AWQ/INT8 production model for cost/latency
- Track “accuracy delta due to quantization” as a first-class metric.

Cross-reference: [[quantization]]

---

## Difficulty bias, curriculum effects, and inference stability

### Difficulty bias in GRPO and its downstream effects
A described GRPO term that normalizes by reward stddev can bias learning toward very easy or very hard prompts (rewards near all-1 or all-0). Some approaches remove reward scaling to mitigate this.

Why it matters for serving:
- Training instabilities can manifest as inference pathologies:
  - nonsensical reasoning traces
  - distribution shift in length
  - brittle performance on certain difficulty bands

Cross-reference: [[reinforcement_learning]]

### “Language mixing” failure mode during RL (serving relevance)
A reported RL failure mode:
- On very hard prompts where **all rollouts get reward 0**, training can destabilize; symptoms included “language mixing” in chain-of-thought and exploding KL/grad norms.
- A mitigation was to add more easy/medium problems.
- Another mitigation: online filtering/curriculum (as in DAPO) that drops too-easy and too-hard prompts.

Serving relevance:
- If a model exhibits such behaviors, they may appear as:
  - sudden incoherent outputs for certain inputs
  - non-English interleavings
- For production, consider:
  - input difficulty heuristics
  - safety rails: language detectors, max-length limits, forced output format
  - fallback to a more stable model when signals trigger

Cross-reference: [[monitoring]], [[safety]] (if these pages exist; otherwise create later)

---

## Practical serving guidance for Maj@K reasoning workloads

### Recommended evaluation-to-serving checklist
Before shipping a reasoning model that uses multi-sample voting:

- **Fix the prompt** and document it (small wording changes can move Pass@1).
- **Fix decoding**: temperature/top_p, and whether you use nucleus sampling or alternatives.
- **Fix stop conditions** (per model).
- **Sweep token budgets** to find:
  - best Maj@K point
  - best accuracy-per-token point
- **Measure average and tail lengths**; tail dominates latency and batching efficiency.
- **Validate deployed artifact**: quantized vs BF16.
- **Track both Pass@1 and Maj@K** (and do not over-optimize one at the expense of the other).

Cross-reference: [[evaluation]], [[prompting]], [[quantization]]

### Compute/cost framing
A useful cost model for serving:
- Total tokens ≈ `num_prompts × num_rollouts × avg_generated_tokens`
- Latency ≈ dominated by max generated tokens in the rollout group + engine overhead

Efficiency levers:
- reduce average length (training and/or decoding)
- reduce tail length (stopping rules, truncation policies)
- increase batching efficiency (length bucketing)
- reduce K when not needed (adaptive sampling)

---

## Open questions / areas to extend

This page should be extended with:
- A concrete **reference architecture** for serving Maj@K:
  - request router
  - rollout workers (vLLM)
  - vote/aggregation service
  - caching of rollouts
- Standardized definitions and formulas for:
  - Pass@1
  - Maj@K
  - “token budget”
  - “test-time scaling economy”
- Guidance on:
  - KV cache memory sizing for long-context
  - throughput modeling under heavy-tailed lengths
  - deterministic vs stochastic decoding in regulated settings

Cross-reference: [[vllm]], [[evaluation]], [[capacity_planning]] (if exists)

---

## Source notes

Integrated from a detailed April 2025 write-up describing training and evaluation of 7B and 14B DeepSeek-R1-distill-based math reasoning models using:
- SFT followed by GRPO variants
- long-context rollouts (up to 32K)
- majority vote metrics (Maj@32)
- engineering stacks including veRL/Open-R1, and vLLM-based rollouts
- merge-based final checkpoints and quantized deployment artifacts