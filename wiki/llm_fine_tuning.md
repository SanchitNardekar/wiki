---
slug: llm_fine_tuning
sources:
- hav4ik.github.io
tags: []
title: LLM Fine-Tuning
updated: '2026-04-14'
---

# LLM Fine-Tuning

LLM fine-tuning is the process of adapting a pretrained large language model (LLM) to improve performance on specific tasks, domains, or behaviors. Modern fine-tuning workflows often combine:

- **Supervised fine-tuning (SFT)**: training on labeled examples (often prompt → response).
- **Reinforcement learning (RL) fine-tuning**: optimizing a policy to maximize a reward signal (e.g., correctness, preference, style), including RLHF-style setups and outcome-based RL for reasoning.

This page summarizes practical lessons and failure modes observed in long-context reasoning model training (math olympiad-style), based on experiments fine-tuning DeepSeek-R1 distilled models using SFT plus GRPO.

Related pages (add as you expand the wiki): [[rlhf]], [[lora]], [[grpo]], [[ppo]], [[dataset_curation]], [[evaluation]], [[model_merging]], [[long_context]], [[reasoning_models]].

---

## Why fine-tune? (Reasoning/math case study)

A key motivation for fine-tuning is to push performance on hard reasoning benchmarks (e.g., AIME-style problems) without training from scratch.

From a practical reproduction effort building on **DeepSeek-R1 distilled models**:

- A team fine-tuned **7B** and **14B** math reasoning models starting from **DeepSeek-R1-Distill-Qwen** baselines.
- Approach: **SFT first**, then **RL (GRPO)**, and finally **model merging** (SFT + GRPO checkpoints).
- Reported gains (AIME’25, majority voting):
  - **14B**: **75.8% Maj@32** (reported as **+8.7%** improvement over baseline).
  - **7B**: **65.8% Maj@32** (reported as **+7.5%** improvement).

These results emphasize a common fine-tuning pattern: **use SFT for skill acquisition**, then **RL for behavior shaping/steering** (e.g., correctness vs verbosity trade-offs).

---

## Core fine-tuning stages

### Stage 1 — Supervised Fine-Tuning (SFT)

SFT adapts a base model to better imitate desirable outputs (e.g., correct math solutions with good reasoning traces).

Practical notes from the case study:

- **Base models**: DeepSeek-R1-Distill-Qwen-7B and -14B.
- **Context length used for SFT**: **16K**.
- **Epochs**: **6**.
- Training infrastructure example: **8× H100 node**.
- Observed behavior:
  - Longer SFT can improve accuracy, but may also cause **unnecessarily long chains-of-thought (CoT)**.
  - A specific issue noted: “the longer we trained, the longer the model’s generated solutions became.”

Cross-reference: [[sft]], [[long_context]].

---

### Stage 2 — Reinforcement Learning (RL) fine-tuning

RL fine-tuning optimizes the model against a reward. In reasoning settings, reward can be:

- **Outcome reward**: did the final answer match the correct one?
- (Optionally) shaping penalties/bonuses: e.g., discouraging overlong generations.

In the case study, the main RL method was **GRPO**, a lightweight policy-gradient variant.

Cross-reference: [[reinforcement_learning]], [[grpo]], [[ppo]].

---

## GRPO in practice (and why it’s used)

### What is GRPO?

GRPO (Group Relative Policy Optimization) is used as a cheaper alternative to PPO:

- It **removes the critic/value network**, reducing memory and training time (reported roughly **~half** compared to PPO).
- Advantage estimates come from **group statistics** over multiple sampled outputs per prompt.

This makes GRPO attractive for budget-limited fine-tuning on long-form reasoning where rollouts are expensive.

Cross-reference: [[grpo]], [[ppo]].

---

## Known GRPO pitfalls (and fixes)

### 1) Length bias / “length hacking”

Observed issue:

- During SFT and/or GRPO, models may trend toward **longer and longer solutions**, sometimes without accuracy gains.
- The REINFORCE++ paper is cited as suggesting length hacking is **GRPO-specific** relative to other algorithms (PPO, RLOO, REINFORCE++), although GRPO can converge faster.

Root cause described (implementation-level):

- If loss is **normalized per-sample** (averaging within each sequence before aggregating across the group), then **tokens in longer sequences contribute less**, implicitly encouraging longer outputs because errors are “diluted.”

Fix suggested by DAPO / Dr. GRPO-style adjustments:

- **Remove per-sample loss normalization** and compute loss **uniformly across all tokens**.

Cross-reference: [[grpo]], [[reward_modeling]], [[training_stability]].

---

### 2) Difficulty bias (advantage normalization)

Issue described:

- A normalization term proportional to  
  \[
  \frac{1}{\mathrm{std}(\{R(q,o_1), \ldots, R(q,o_G)\})}
  \]
  can bias training toward **very easy** or **very hard** prompts (where rewards collapse near all-1 or all-0).

Mitigation (Dr. GRPO-style):

- **Remove reward scaling** entirely (i.e., don’t scale advantage by reward std).

---

### 3) Do we need KL regularization?

In RL fine-tuning, KL penalties constrain divergence from a reference policy.

Notes from the case study:

- GRPO uses **forward KL** \(D_{KL}(\pi_\theta \| \pi_\text{ref})\) (contrast: PPO often uses reverse KL in some RLHF implementations).
- Reported: forward vs reverse KL made **negligible difference** in small experiments (1.5B), though others reported worse results with forward KL.
- DAPO is described as going further and **removing KL entirely**, arguing that in long-CoT reasoning, models diverge enough that KL regularization may not help.

**Potential contradiction / tension to track in the wiki:**
- Some practitioners view KL as essential for stability and avoiding reward hacking, while this source suggests **KL may be unnecessary** (at least in some long-CoT regimes). This should be evaluated per setting and reward design.

Cross-reference: [[kl_divergence]], [[rlhf]], [[training_stability]].

---

## Dataset curation for fine-tuning (reasoning)

Fine-tuning quality depends heavily on data selection—especially for reasoning.

### Data goals noted

- Focus on **high-quality reasoning traces** with controlled length:
  - Collect traces under **16K tokens**.
  - Rationale: many correct traces were under ~6K; 16K balanced accuracy and compute.

### Example curation pipeline (math reasoning)

A multi-step filtering pipeline was described:

1. **Initial pool**: filter math word problems from NuminaMath-1.5 across:
   - Algebra, Geometry, Number Theory, Combinatorics
   - Sources include olympiads, AoPS forums, AMC/AIME, references, number theory sources
2. **Join with correct R1 traces**: from OpenR1-Math-220k
   - Filtering reduced ~800K problems to **~27K**
3. **Difficulty filtering via sampling**:
   - Sample multiple solutions per problem (e.g., **8** solutions)
   - Remove “easy” problems by keeping only those with **≤ 7 correct solutions**
   - Reduced to **~8K** problems
4. Add a subset of **Light-R1 stage 2** data (after removing >16K CoT), de-duplicate, then difficulty filter further to **~2K**
5. Final SFT dataset: **>10K samples**
   - ~8K from filtered NuminaMath-1.5
   - ~2K from Light-R1 subset
6. Harder half used for RL stage (GRPO).

### Data sources explicitly avoided (and why)

- Avoided **cn_k12** (reported lower difficulty; harmed hard-problem performance).
- Avoided synthetic datasets (e.g., **Orca-Math**) because:
  - Often generated by weaker LLMs with weaker validators.
  - Claimed to be more useful for training reasoning from scratch than for fine-tuning an already-strong reasoning model.

Cross-reference: [[dataset_curation]], [[data_quality]], [[reasoning_traces]].

---

## Long-context considerations in fine-tuning

Long-context RL is substantially harder than short generations (e.g., GSM8K-like <1K tokens).

### Practical observations

- RL at **8K/16K/24K** token traces introduces major systems challenges:
  - VRAM pressure and GPU utilization bottlenecks
  - idle time caused by “waiting for the longest sequence”
  - need for sequence packing and efficient rollout engines (e.g., vLLM)

### Iterative context lengthening: doesn’t always transfer

A technique used in some small models is to train progressively longer contexts (8K→16K→24K). However:

- In this case study, **training at shorter contexts significantly reduced accuracy** when evaluating at longer inference lengths (for 7B/14B).
- Training directly at the target long context (e.g., **16K from the start**) achieved:
  - similar shortening of solution length
  - **better accuracy** (Pass@1 and Maj@32) than “start short then extend”

**Important nuance:** This is presented as scale-dependent:
- May work for ~1.5B models (per comparison to DeepScaleR/DeepCodeR observations),
- but not reliably for 7B/14B.

Cross-reference: [[long_context]], [[curriculum_learning]].

---

## LoRA vs Full Fine-Tuning (FFT) in RL

LoRA can reduce memory and improve iteration speed, but integration with distributed training can be complex.

### Reported findings

- For 7B GRPO:
  - **LoRA converged faster** than FFT and was **more VRAM-efficient**.
- For 14B:
  - LoRA sometimes converged faster in some runs, but:
    - best model reportedly still trained with **FFT**
    - confounded by differences beyond LoRA-vs-FFT toggles
    - no budget for clean ablations; possible implementation bugs cannot be ruled out

### Engineering caution

- “LoRA is easy” becomes false with:
  - **FSDP**, DTensor/dtype mismatch pitfalls
  - **offloading + vLLM rollouts**
  - need to support both base and adapter forward passes (actor vs ref)
  - risk of VRAM spikes during adapter merging and vLLM startup

Cross-reference: [[lora]], [[fsdp]], [[distributed_training]].

---

## Model merging (SFT + RL checkpoints)

Model merging can combine benefits of different training phases.

### Example method

- Used **MergeKit** with **TIES** merge method.
- Used simple hyperparameters (weights=1, density=1) without extensive tuning.

### Observed behavior

- **7B**: merging improved overall performance:
  - merged model surpassed both SFT and GRPO checkpoints on **accuracy and token economy**
- **14B**: merging was “more of a compromise”
  - suggests trade-offs become sharper at larger scale or that checkpoints diverge more

Cross-reference: [[model_merging]], [[mergekit]].

---

## Evaluation practices for fine-tuned reasoning models

### Uncontaminated benchmarks

To reduce data contamination concerns:

- Evaluated on **AIME 2025**, released after DeepSeek R1 training.
- Data source NuminaMath-1.5 was collected beforehand (claimed to reduce leakage risk).

Also used a small local validation set (“CV”):

- **40 problems** (AIME’25 + AIMO2 reference problems)
- used for tracking progress across stages

Cross-reference: [[evaluation]], [[benchmark_contamination]].

---

### Metrics: Pass@1 vs Maj@K (majority voting)

The source strongly emphasizes:

- **Better Pass@1 does not imply better Maj@32.**
- With larger token budgets, Maj@32 can **get worse**, even if Pass@1 improves.

Reported phenomenon:

- With too much “thinking time” (larger max_len / token budgets), models may:
  - self-doubt
  - drift
  - produce more wrong answers despite more reasoning

This is an important evaluation caveat for fine-tuning and inference-time scaling:
- Always specify:
  - sampling temperature/top-p
  - max_len / token budget
  - stop conditions
  - number of rollouts
  - how majority voting is computed and averaged

Cross-reference: [[evaluation]], [[test_time_scaling]].

---

## Training stability failure mode: language mixing

A notable RL failure mode observed:

- During GRPO training, some runs began mixing languages mid-CoT (e.g., switching to Chinese).
- It correlated with:
  - exploding KL divergence and gradient norms
  - training instability triggered by prompts where **all rollouts received reward=0**
- Fix suggested:
  - include more easy/medium problems to stabilize learning signal
  - use online filtering/curriculum (e.g., DAPO) to avoid too-hard batches

Cross-reference: [[training_stability]], [[curriculum_learning]].

---

## Compute and cost considerations

Fine-tuning long-context reasoning models is expensive, but still far cheaper than pretraining.

Examples given:

- A DeepScaleR-1.5B run: **~$5,000** for ~10% gain (as cited in the narrative).
- Scaling that approach to 7B/14B was described as infeasible under a hobby budget.
- A claimed end-to-end run (SFT + GRPO) for final 14B model: **< $800**.

**Note:** Costs vary dramatically with:
- sequence length (8K vs 16K vs 24K)
- rollout count per prompt
- cluster type (H100 vs H200), utilization, and framework efficiency
- whether using [[lora]] vs full fine-tuning
- batching/packing, reuse strategies (“data echoing”-like reuse of rollouts)

Cross-reference: [[training_infrastructure]], [[cost_modeling]].

---

## Practical hyperparameter themes (from long-context GRPO)

Open questions noted:

- Given a fixed tokens-per-update budget, should you use:
  - more prompts per batch with fewer rollouts each, or
  - fewer prompts with more rollouts per prompt?
- Trade-off described:
  - More rollouts helps hard questions find a rewarded trajectory
  - Too few prompts risks overfitting/biasing gradients to a small set

Rules of thumb reported:

- rollouts per problem: **8–16** (dropping to 6 hurt performance)
- prompts per batch: **12–24** (below 10 hurt reward curves)
- learning rate: **1e-6 to 4e-6** (higher like 5e-6 unstable in some 1.5B runs)
- batch reuse (rollout reuse) can work without obvious degradation in some settings, but may amplify small-batch bias

Cross-reference: [[hyperparameters]], [[training_stability]].

---

## Contradictions / tensions to monitor

Since this is a new page, contradictions are mostly internal tensions from the source that should be tracked as the wiki grows:

- **“GRPO is less effective than PPO”** (stated as an expectation) vs **“GRPO converges faster”** and is widely used due to efficiency.  
  - Practical takeaway: effectiveness depends on reward design, stability tweaks, and implementation.
- **KL regularization**: some evidence suggests KL choice (forward vs reverse) is negligible in some experiments, and some approaches remove KL entirely—contrary to more conservative RLHF practice where KL is central for stability.
- **Iterative context lengthening**: sometimes reported as useful (in some projects/smaller models) but **harmful at 7B/14B** when targeting long-context inference.

As additional sources are integrated, this section should explicitly compare claims across sources and note which settings (model size, task, reward type, context length) explain the disagreement.

---

## Summary checklist: building an effective fine-tuning pipeline (reasoning)

- **Data**
  - Curate for difficulty and correctness; avoid low-quality synthetic traces for already-strong reasoning models.
  - Control trace length (e.g., cap at 16K) to manage compute and verbosity.
- **SFT**
  - Use SFT for skill acquisition; watch for “longer training → longer CoT”.
- **RL (GRPO)**
  - Expect GRPO-specific pitfalls (length bias, difficulty bias).
  - Consider DAPO/Dr.GRPO-style adjustments (loss normalization, reward scaling changes, online filtering).
- **Long-context**
  - Train at intended inference context length when possible; short-context pretraining may not transfer at larger scales.
- **Evaluation**
  - Report Pass@1 and Maj@K; don’t assume they track together.
  - Specify token budgets and sampling configs; larger budgets can reduce Maj@K.
- **Engineering**
  - Framework and rollout engine choice matters for utilization and feasibility (sequence packing, collocation, offloading).
- **Merging**
  - Merging SFT and RL checkpoints can improve results (especially at smaller scales), but may be a compromise at larger scales.

Cross-reference: [[dataset_curation]], [[sft]], [[grpo]], [[lora]], [[model_merging]], [[evaluation]], [[long_context]].