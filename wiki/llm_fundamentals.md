---
slug: llm_fundamentals
sources:
- hav4ik.github.io
tags: []
title: LLM Fundamentals
updated: '2026-04-14'
---

# LLM Fundamentals

This page collects foundational concepts for understanding and improving large language models (LLMs), with an emphasis on **reasoning models** and practical training/evaluation lessons drawn from recent open-weight math-reasoning work (notably DeepSeek-R1 distilled models fine-tuned via SFT + GRPO).

---

## 1) What are “reasoning” LLMs?

**Reasoning models** are LLMs tuned to solve tasks that require multi-step inference (e.g., math olympiad problems) and often produce long intermediate “reasoning traces” (a.k.a. chains-of-thought, CoT).

Key points:

- Reasoning performance is often benchmarked on math contest datasets such as **AIME** (American Invitational Mathematics Examination) and competition-style suites like Kaggle’s **AIMO2**.
- Open-weight reasoning progress accelerated after the release of **DeepSeek R1**, which provided both:
  - a strong model (comparable in positioning to OpenAI’s o1 per the source narrative), and  
  - a paper describing its algorithmic recipe—enabling reproductions and improvements.

Cross-references:
- Related: [[reinforcement_learning]], [[rlhf]], [[sft]], [[evaluation]], [[prompting]]

---

## 2) Benchmarks and why they matter

### AIME as a reasoning benchmark
- AIME problems are “hard” because they require careful multi-step reasoning and have many corner cases.
- Observed failure modes resemble human errors:
  - Example: misreading problem constraints (e.g., confusing whether an empty subset is allowed when selecting subsets).

### AIMO2 (Kaggle) and “AI-hard” problems
- AIMO2 is described as a Kaggle competition with a **$2,000,000** prize pool aimed at advancing open-source reasoning models.
- The public reference set includes problems intended to be “AI hard,” often featuring:
  - many corner cases
  - geometry (noted as difficult due to lack of native visual understanding)

Cross-references:
- Related: [[benchmarks]], [[math_reasoning]], [[test_time_scaling]]

---

## 3) Training pipelines for reasoning models

A common practical recipe for improving distilled reasoning models is:

1. **Stage 1 — Supervised Fine-Tuning (SFT)** on curated high-quality reasoning traces  
2. **Stage 2 — Reinforcement Learning (RL)** to further improve correctness and/or steer behaviors (e.g., shorter solutions)

This is sometimes framed as:

- **Skill acquisition** via SFT  
- **Behavior steering** via RL

Cross-references:
- Related: [[sft]], [[reinforcement_learning]], [[rlhf]]

---

## 4) GRPO: a lightweight RL method used for reasoning

**GRPO (Group Relative Policy Optimization)** is used as a main RL method in the source. It is described as a lightweight policy-gradient approach introduced in DeepSeek Math.

Main characteristics:

- Similar “family” to PPO-style methods but:
  - **omits the critic/value network**, reducing memory usage and training time (reported ~half in the source)
  - computes advantage using **group statistics** over multiple samples/rollouts

Practical note:
- The ecosystem for long-context RL has evolved quickly; frameworks that initially struggled with rollouts and colocation later added features to support long-context GRPO.

Cross-references:
- Related: [[ppo]], [[policy_gradients]], [[grpo]], [[long_context_training]]

---

## 5) Known pitfalls in RL for LLM reasoning (and mitigations)

### 5.1 Length bias (“length hacking”)
Observed phenomenon:
- During training, models may learn to produce **longer and longer** reasoning traces.

Source claims and hypotheses:
- Verbosity is **not necessarily** an emergent requirement of reasoning quality.
- Some works (e.g., REINFORCE++ per the source) report that **GRPO specifically** can incentivize length growth compared to other algorithms.

Mitigation described:
- Two papers (DAPO and “Dr. GRPO”) independently identify an **implicit length bias** in the original GRPO formulation:
  - longer incorrect solutions can be penalized less on a token-wise basis due to **per-sample loss normalization**
- Implementation-level fix:
  - remove per-sample normalization and compute loss uniformly across all tokens

Cross-references:
- Related: [[grpo]], [[reward_modeling]], [[optimization]]

### 5.2 Difficulty bias from advantage/reward normalization
Source notes that a normalization term based on the standard deviation of rewards can bias training toward:

- extremely easy questions (all rewards ~1)
- extremely hard questions (all rewards ~0)

Mitigation described:
- “Dr. GRPO” approach removes reward scaling entirely.

Cross-references:
- Related: [[curriculum_learning]], [[online_sampling]]

### 5.3 Do we even need KL regularization?
Source discussion highlights:

- GRPO uses **forward KL** $D_{KL}(\pi_\theta\|\pi_{\text{ref}})$, while PPO often uses **reverse KL**.
- Empirical observations in the source are mixed:
  - Some report forward KL worse
  - The author’s experiments found negligible difference on at least one smaller model
- DAPO (as described) argues KL may be removed entirely in long-CoT regimes because the model diverges enough that regularization is less helpful.

**Potential contradiction to watch for (future sources):**
- Many RLHF recipes treat KL as essential for stability; this source suggests KL can sometimes be reduced/removed. If other internal pages or sources assert KL is always required, note this as a context-dependent disagreement.

Cross-references:
- Related: [[kl_divergence]], [[rlhf]], [[regularization]]

---

## 6) Data curation fundamentals for reasoning SFT/RL

The source emphasizes that with limited budget, **data quality and filtering** can matter more than “pure RL scaling.”

Key dataset curation principles described:

- Focus on **high-quality solution traces** with an upper bound (e.g., **≤16K tokens**).
  - Rationale: most correct base-model traces are reportedly under ~6K; 16K is a compute/accuracy tradeoff.
- Start from a large pool (e.g., math word problems across algebra/geometry/number theory/combinatorics).
- Join problems with **known-correct reasoning traces** from a stronger teacher/source.
- **Difficulty filtering** via sampling multiple solutions per problem:
  - keep problems that are not “too easy” (e.g., not too many correct sampled solutions)
- Avoid certain sources if they degrade hard-problem performance:
  - “low difficulty” sources can harm performance on harder benchmarks
  - synthetic datasets generated/validated by weaker models may be less useful for fine-tuning an already strong reasoning model (source claim)

Cross-references:
- Related: [[data_curation]], [[synthetic_data]], [[distillation]], [[curriculum_learning]]

---

## 7) Long-context RL engineering fundamentals

Long-context RL (8K–24K+ reasoning traces) adds distinct engineering constraints beyond short-answer RL.

### 7.1 Framework capabilities that matter
Capabilities called out as important for long-context RL:

- FSDP / sharding + CPU offloading for memory
- sequence packing and long-sequence attention strategies
- rollout engines (e.g., vLLM-based rollouts)
- hybrid/collocated training to reduce VRAM duplication between actor/reference/rollout engines

The source notes that framework support changes quickly; comparisons can become outdated.

Cross-references:
- Related: [[fsdp]], [[vllm]], [[distributed_training]], [[systems]]

### 7.2 Training bottlenecks: rollouts dominate
- Rollout generation can take longer than optimization steps.
- A major inefficiency: “tail latency” waiting for the longest sequence in a batch → idle GPUs.

Mitigations described:

- increase problems/rollouts per step to reduce idle time proportion
- reuse rollouts with multiple optimization steps (similar to “data echoing”); the author reports no drop in performance in one setup

Cross-references:
- Related: [[throughput_optimization]], [[batching]], [[sampling]]

### 7.3 LoRA is not “free” in RL systems
While LoRA is popular for SFT, integrating LoRA into long-context RL with sharding/offloading/rollout engines can be tricky:

- dtype mismatches and DTensor/sharding interactions
- needing both base + adapter forward passes for reference computations
- sharded weight merge + transfer into a rollout engine can cause VRAM spikes and unstable memory behavior

Cross-references:
- Related: [[lora]], [[qlora]], [[parameter_efficient_finetuning]]

---

## 8) Model merging as a practical tool

The source describes merging SFT and GRPO checkpoints using MergeKit with a TIES merging method (with simple hyperparameters).

Observed effects:

- For a 7B model, merging improved both:
  - accuracy
  - token economy (shorter outputs for similar/better correctness)
- For 14B, merging was “more of a compromise” (suggesting tradeoffs may be scale-dependent)

Cross-references:
- Related: [[model_merging]], [[mergekit]]

---

## 9) Evaluation fundamentals for reasoning models

### 9.1 Contamination-aware evaluation
The source evaluates on **AIME 2025** specifically because it was published after DeepSeek R1 was trained, aiming to treat it as “uncontaminated.”

Takeaway:
- Prefer benchmarks released after base-model training and after dataset collection, when possible.

Cross-references:
- Related: [[data_contamination]], [[evaluation]]

### 9.2 Pass@1 vs Majority Voting (Maj@K)
The source uses:

- **Pass@1**: accuracy of a single sample
- **Maj@32**: majority vote accuracy over 32 sampled traces (with repeated simulations to reduce noise)

Important claim:
- **Better Pass@1 does not necessarily mean better Maj@32.**

Additionally, the source reports a surprising effect:

- At very large token budgets (e.g., 32K), **Maj@32 can get slightly worse than at 16K** because the model may “self-doubt” or overthink and drift into wrong answers.

This is a key “LLM fundamentals” lesson:

- More “thinking time” (more tokens) is not monotonically better for aggregate accuracy.

Cross-references:
- Related: [[test_time_scaling]], [[self_consistency]], [[sampling]]

### 9.3 Stopping conditions can change results
The source reports using a custom stop condition for their fine-tuned models because they were trained with more restricted generation lengths and often produced correct answers before finishing a full chain.

But:
- Applying the same stop condition to baseline DeepSeek-R1 models reportedly reduced accuracy, so baselines were evaluated with default EOS stopping.

Fundamental point:
- **Evaluation settings (prompt, stop rules, max tokens) materially affect results** and can create misleading comparisons if not standardized.

Cross-references:
- Related: [[evaluation]], [[prompting]], [[generation_parameters]]

### 9.4 Prompting differences can look like model differences
The author notes an inability to reproduce an external reported Pass@1 for a 32B model, hypothesizing:

- different prompt that hints the answer format (e.g., “answer should be an integer”) can boost Pass@1
- larger models may follow system prompts better, amplifying prompt sensitivity

Fundamental point:
- When comparing models, control prompts; otherwise you may be measuring prompting rather than weights.

Cross-references:
- Related: [[prompting]], [[reproducibility]]

---

## 10) Practical observations about scaling and transfer

### 10.1 Tricks that work at small scale may not transfer
The source describes **iterative context lengthening** (train at 8K then 16K, etc.):

- worked on small models (e.g., 1.5B in some projects)
- **did not translate well to 7B/14B**; training on shorter contexts harmed performance at the intended longer inference length

Takeaway:
- If your target inference context is long (e.g., 16K), training at that context from the start may preserve accuracy better.

Cross-references:
- Related: [[scaling_laws]], [[long_context_training]]

### 10.2 Language mixing as an RL failure mode
A striking RL failure described:

- During GRPO training, a model started mixing languages mid-reasoning.
- It coincided with:
  - exploding KL divergence and gradient norms
  - hard prompts where **all rollouts received reward 0**
- Fix suggested:
  - add more easy/medium problems to stabilize learning signal
- DAPO-style online filtering helped avoid the issue by dynamically filtering too-easy/too-hard prompts, functioning like an online curriculum.

Takeaway:
- RL stability can depend heavily on **reward diversity** and **batch difficulty composition**.

Cross-references:
- Related: [[reward_design]], [[curriculum_learning]], [[training_stability]]

---

## 11) Case study snapshot: improving DeepSeek-R1-distilled math models

The source describes a team training **7B and 14B** models from DeepSeek-R1-Distill bases using SFT + GRPO, reporting improvements on AIME’25 Maj@32:

- 14B: **75.8% Maj@32** (reported +8.7% improvement over baseline)
- 7B: **65.8% Maj@32** (reported +7.5% improvement)

Additional claims:

- their 14B result reportedly surpasses a larger 32B distilled baseline in that setting
- an end-to-end (SFT+GRPO) run cost reported as **< $800** (for their final 14B model)
- they observed quantization could erase a sizable portion of training gains (important for deployment)

Cross-references:
- Related: [[cost_modeling]], [[quantization]], [[deployment]]

---

## 12) Open questions and tensions (to track as you learn)

The source explicitly or implicitly raises several unresolved/nuanced points:

- **Token budget vs accuracy** is not monotonic (more tokens can reduce Maj@K due to overthinking/self-doubt).
- **KL regularization** may be optional or harmful in some long-CoT regimes (source suggests DAPO removes KL), but many RLHF recipes rely on KL for stability.
- **LoRA vs full fine-tuning (FFT)**:
  - LoRA may converge faster and be more memory-efficient
  - but the best run in the source narrative used FFT, and the author notes lack of controlled ablations
- **Batch design tradeoff**:
  - more rollouts per prompt improves odds on hard tasks
  - more prompts per batch may generalize better; too few prompts biases gradients

These should be treated as **context-dependent** rather than universal laws.

Cross-references:
- Related: [[ablation_studies]], [[reproducibility]], [[optimization]]

---

## References (source integrated)

- Chan Kha Vu, “Improving DeepSeek R1 in Math” (Apr 18, 2025): training distilled DeepSeek-R1 7B/14B with SFT + GRPO; discussions of GRPO pitfalls (length/difficulty bias), DAPO/Dr.GRPO ideas, long-context RL engineering, evaluation methodology (AIME’25, Maj@32 vs Pass@1), and practical training lessons.