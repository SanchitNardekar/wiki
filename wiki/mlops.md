---
slug: mlops
sources:
- hav4ik.github.io
tags: []
title: MLOps and ML Infrastructure
updated: '2026-04-14'
---

# MLOps and ML Infrastructure [[mlops]]

MLOps/ML Infrastructure is the set of practices, systems, and engineering disciplines used to **develop, train, evaluate, deploy, and operate machine learning models reliably at scale**. While often discussed in the context of classic ML and short-form generation, modern LLM and reasoning-model training introduces distinctive infra requirements: **long-context training**, **reinforcement learning (RL) pipelines**, **test-time scaling evaluation**, and **tight cost/throughput constraints**.

This page summarizes practical MLOps/infra learnings from a real project that improved **DeepSeek-R1-Distill** math reasoning models (7B and 14B) via **SFT** and **GRPO** (an RL method), including dataset curation, training/eval design, and systems bottlenecks.

> Source context: a small self-funded team trained 7B/14B math reasoning models based on DeepSeek-R1-Distill using SFT + GRPO; their 14B reached **75.8% Maj@32 on AIME’25** (+8.7%), and 7B reached **65.8% Maj@32** (+7.5%).  
> This is a strong example of how ML infrastructure decisions directly shape achievable model quality per dollar.

---

## Scope: what “ML infrastructure” covers

For LLM/reasoning-model workflows, infra typically includes:

- **Data systems**
  - Sourcing, filtering, deduplication, contamination checks
  - Trace/trajectory storage (e.g., reasoning traces, rollouts)
  - Dataset versioning and provenance
- **Training systems**
  - Distributed training (FSDP/ZeRO, tensor parallelism)
  - Fine-tuning (full fine-tune, LoRA/QLoRA)
  - RL training loops (rollout engine + trainer + reward computation)
  - Long-context memory optimization (packing, attention/memory tricks)
- **Evaluation systems**
  - Benchmarks and local “CV” sets
  - Metrics beyond Pass@1 (e.g., Maj@K, token economy)
  - Reproducible sampling and aggregation
- **Deployment & inference**
  - Quantization impacts (AWQ etc.)
  - Serving engines (e.g., vLLM) and rollouts
  - Token budget management and stopping conditions
- **Experiment operations**
  - Cost tracking, run scheduling, failure handling
  - Artifact management (checkpoints, merges, configs)
  - Observability (GPU utilization, KL/grad norms, reward stats)

Cross references (create/relate as appropriate):
- RLHF / RL training: [[rlhf]]
- Evaluation and metrics: [[evaluation]]
- Data curation: [[datasets]]
- Distributed training: [[distributed-training]]
- Model fine-tuning: [[fine-tuning]]
- Parameter-efficient tuning: [[lora]]
- Inference/serving: [[inference]]

---

## Example end-to-end pipeline (SFT → RL → merge → eval)

A practical pipeline used to improve DeepSeek-R1-Distill math models:

1. **Dataset curation** (high-quality long-form solution traces, capped length)
2. **Stage 1: Supervised Fine-Tuning (SFT)** on curated traces at long context
3. **Stage 2: Reinforcement Learning** with GRPO (long-context rollouts)
4. **Model merging** of SFT + GRPO checkpoints (TIES via MergeKit)
5. **Evaluation** on uncontaminated benchmark + local CV, using Maj@K and token budgets

Key infra implication: RL for reasoning models is not a “small add-on”; it is a **separate distributed system** with different bottlenecks than SFT.

---

## Data infrastructure: curation, filtering, and contamination avoidance

### Curation goals
- Focus on **harder problems** in Algebra/Geometry/Number Theory/Combinatorics.
- Prefer **high-quality reasoning traces** with controlled length.
- Optimize for both **accuracy** and **compute efficiency** (training + inference).

### Length management policy
- Collected solution traces **under 16K tokens**.
  - Rationale: most correct DeepSeek R1 outputs are under ~6K; 16K was a workable compromise between accuracy and compute cost for 7B/14B.

### Multi-stage filtering workflow (example)
- Start with a large pool (e.g., **NuminaMath-1.5** math word problems).
- Join with correct reasoning traces from **OpenR1-Math-220k**.
  - Example reduction: **800K → 27K** after joining/filters.
- Difficulty filtering via sampling multiple rollouts per problem and removing easy items:
  - Sample **8 solutions per problem** (max_len 8K) using a 7B AWQ model and keep only those with **≤7 correct solutions** (i.e., not too easy).
  - Similar filtering with a 14B AWQ model to find harder problems for RL.
- Add selected subsets of **Light-R1 stage 2 data**, remove duplicates, then re-filter by difficulty.
- Final SFT dataset size: **~10K samples** (e.g., 8K + 2K), with the **harder half used for GRPO**.

### Explicitly avoided sources (quality control)
- Avoided a low-difficulty source (cn_k12) because it **hurt performance on harder problems**.
- Avoided synthetic math datasets (e.g., Orca-Math) due to concerns:
  - Often created by weaker LLMs with weaker correctness validation
  - Claimed to be more useful for *training from scratch* than for improving already-strong reasoning models

### Contamination and benchmark hygiene
- Used **AIME 2025** as an “uncontaminated” benchmark, published after DeepSeek R1 training.
- Also noted NuminaMath-1.5 was collected beforehand, reducing leakage risk.

Infra takeaways:
- Dataset pipelines should support:
  - **Join operations** between problems and trace corpora
  - **Difficulty estimation via sampling**
  - **Dedup + length filtering**
  - **Provenance tracking** (what came from where, and when)

See also: [[datasets]], [[data-quality]], [[benchmarking]].

---

## Training infrastructure: SFT at long context

### SFT training setup (example)
- Base models: **DeepSeek-R1-Distill-Qwen-7B** and **14B**
- Hardware: **8×H100 node**
- Context length: **16K**
- Epochs: **6**
- Observation: longer SFT can improve accuracy but may cause **unnecessarily long chains-of-thought** (CoTs).

Infra implications:
- Long-context SFT needs:
  - Stable distributed training configs
  - Memory-efficient attention & activation checkpointing strategies (implementation-dependent)
  - Logging of **sequence lengths over time** to detect length drift

Related: [[fine-tuning]], [[distributed-training]].

---

## RL infrastructure: GRPO at long context is a different beast

### Why GRPO
GRPO is described as a “lightweight” policy gradient variant:
- Removes the **critic network** (vs PPO), cutting memory usage and training time “roughly in half” (per source).
- Used in DeepSeek R1-style reasoning-model RL pipelines.

Related: [[rlhf]], [[reinforcement-learning]].

### Contradictions / nuances to note
- The source says GRPO is “arguably less effective than PPO” *but* more compute/memory efficient.
- It also reports GRPO can suffer from **length bias** (“length hacking”), which some papers argue is **GRPO-specific**, while other algorithms (PPO, RLOO, REINFORCE++) may avoid the same output-length explosion at similar KL budgets.
- Net: “GRPO is cheaper” and “GRPO converges fast” can both be true, while “GRPO has unique failure modes” is also true.

(There is no prior page content to contradict; this is noted as an internal tension in the new source.)

---

## Known GRPO failure modes and infra-level mitigations

### 1) Length bias (“length hacking”)
Observed phenomenon:
- SFT-distilled DeepSeek-R1 models tend to generate **longer and longer solutions** over training.
- GRPO can amplify this due to formulation/implementation details.

Key claim from source:
- Length bias can be largely an **implementation issue**: if you normalize losses per-sample before group aggregation, tokens in longer sequences contribute less to overall loss.
- Fix suggested by DAPO / Dr. GRPO: **remove per-sample loss normalization** and compute loss uniformly over tokens.

Infra implications:
- Ensure the RL trainer implementation is audited for:
  - Loss normalization behavior
  - Token-level weighting
  - Max_len truncation effects (can bias reward and gradients)

Related: [[training]], [[rlhf]].

### 2) Difficulty bias from reward scaling
Source notes a term like `1/std(rewards_in_group)` in “original GRPO” can bias toward:
- Very easy or very hard questions (rewards near all-1 or all-0)
Mitigation:
- Dr. GRPO removes reward scaling.

Infra implications:
- Reward normalization/scaling choices are not just “math”; they change which samples dominate training.
- Track reward distribution stats and the fraction of all-zero/all-one groups.

### 3) Do we need KL regularization?
Notes from source:
- GRPO uses **forward KL** $D_KL(π_θ || π_ref)$ (vs PPO’s reverse KL in some RLHF setups).
- Some experiments reported negligible difference, while DAPO argues to remove KL entirely for long-CoT reasoning because the policy diverges enough that regularization may not help.

Infra implications:
- KL is a control knob that trades off:
  - Stability vs exploration
  - Preventing collapse vs enabling divergence
- Log KL and gradient norms; watch for instability signs (see language mixing below).

---

## Practical RL training design choices (7B vs 14B)

### Context length strategy
- For 7B:
  - GRPO performed in two stages: **8K → 16K context**
- For 14B:
  - Training on much shorter contexts **reduced accuracy** at intended inference lengths.
  - Final submission used a merge of SFT+GRPO on **~6K context** to regain accuracy via merging.
  - A 16K-context GRPO run existed late in the competition.

**Contradiction/tension (within the source):**
- Iterative context lengthening (8K→16K) is described as helpful at 1.5B scale (per DeepScaleR), but the team reports it **does not translate well** to 7B/14B and can hurt long-length inference accuracy.  
- Operational takeaway: techniques can be **scale-sensitive**; infra should support rapid A/B at target context lengths.

### LoRA vs full fine-tuning (FFT) in GRPO
- LoRA was found to:
  - Converge faster than FFT in some settings (notably 14B, 8K context experiments)
  - Be more VRAM-efficient
- But:
  - Best 14B GRPO model was still trained with FFT
  - The author explicitly notes lack of budget for clean ablations and potential confounders/bugs

Infra implications:
- Your platform should support both:
  - FFT RL runs (heavy memory)
  - LoRA RL runs (complex integration with FSDP + rollout engine)
- And must enable **controlled ablations** (same seeds, same data slices, same rollout policy lag, etc.) when budget allows.

Related: [[lora]], [[fine-tuning]].

---

## Systems engineering: frameworks, scaling, and bottlenecks

### Framework selection for long-context RL
Key point:
- Running GRPO on short generations (<1K tokens) is easy; **8K–24K** token rollouts change everything.

Source comparison (high-level):
- veRL (HybridFlow) supported:
  - FSDP + CPU offloading
  - Sequence packing + Ulysses (long sequence handling)
  - “Full model collocation” / hybrid engine approaches
- TRL’s GRPOTrainer initially had scaling limitations (e.g., one device per node for actor rollouts), but later added key features (e.g., Ulysses, Dr. GRPO techniques).
- Open-R1 used TRL with a faster GRPOTrainer variant (by a Kaggle user).

Infra takeaways:
- Framework capabilities evolve rapidly; bake in:
  - **Upgrade paths**
  - Compatibility tests
  - Performance regression monitoring
- Choose based on:
  - Long-context support
  - Rollout throughput
  - Memory/offload strategies
  - Ease of integrating new algorithmic tweaks (DAPO, Dr. GRPO)

Related: [[distributed-training]], [[inference]].

### Training bottlenecks: rollouts dominate wall time
Observed in a 14B, 16K-context GRPO run on 8×H200:
- Each global step:
  - Generate 256 samples (32 problems × 8 rollouts)
  - Perform 4 optimization steps
- Rollouts take the longest.
- Idle GPU gaps occur due to waiting for the longest sequence.

Mitigations used:
- Increase number of problems/rollouts per step (reduce idle fraction).
- Reuse rollouts by splitting global batch into minibatches and doing multiple optimization steps (similar to “data echoing”).
  - In one setup, reusing rollouts twice (policy lag up to 4 steps) showed no drop vs standard.

Infra takeaways:
- For RL with variable-length sequences, you need:
  - Better batching/packing
  - Variance-aware scheduling
  - Profiling that separates rollout/offload/train phases

### Asynchronous RL
- Mentioned as a “cleaner solution” (e.g., DeepCodeR), but requires extra nodes.

Infra implication:
- Async RL is an architectural choice that trades **hardware complexity** for **utilization**.

### LoRA + FSDP + rollout engine integration is hard
Key operational difficulties:
- Dtype mismatches with DTensor/gradients when combining LoRA with FSDP1
- Need both base and LoRA forward passes (actor vs ref usage patterns)
- Offloading into a rollout engine (e.g., vLLM) requires sharded weight merging and can cause VRAM spikes due to asynchronous merging overlapping with vLLM startup
- Workaround reported: reduce memory usage on trainer and vLLM sides; add synchronization barriers if needed (at performance cost)

Infra takeaways:
- “LoRA is easy” is true for simple fine-tunes, but can be false in RL:
  - budget time for integration testing and memory spike debugging
  - add explicit VRAM spike detection and automated retries/fallback settings

---

## Evaluation infrastructure: metrics, token budgets, and reproducibility

### Why Pass@1 is not enough
The source emphasizes:
- “Better Pass@1 does not mean better Maj@32.”
- At longer token budgets, models can **self-doubt** and get worse on majority voting.

This is a key MLOps lesson: you must align evaluation with the product/competition objective (e.g., majority vote under budget) rather than defaulting to Pass@1.

Related: [[evaluation]].

### Sampling settings (example)
- Precision: **bfloat16**
- Sampling: temperature=0.75, top_p=0.95
- Collect **64 traces per question** with max_len up to 32768
- For Maj@K:
  - sample K traces from the 64-trace pool
  - repeat aggregation 16 times and average (reduce noise)

Infra takeaways:
- Store raw traces and sampling metadata.
- Provide eval tooling that supports:
  - resampling without regenerating
  - repeated aggregation to estimate noise/variance

### Stopping conditions can change results
- They used `stop=['']` for their models because models were trained with more restricted lengths and often finished early.
- Applying the same stopping condition to baseline DeepSeek-R1 models reduced accuracy; thus baselines used default EOS stopping.

Infra implication:
- Stop conditions are part of the evaluation contract; they must be standardized or explicitly noted in reports.

### Token budget effects: longer isn’t always better
Observation:
- At 32K budget, models performed slightly worse on Maj@32 than at 16K.
- Hypothesis: more room to think → more hallucination/self-doubt.

Infra implication:
- Add “token budget sweeps” to evaluation:
  - 8K, 9K, 12.8K, 16K, 20K… rather than a single max_len.

---

## Model merging as an MLOps primitive

Technique:
- Merge SFT and GRPO checkpoints using **MergeKit** with **TIES** method.
- They used weights=1, density=1 (no hyperparameter tuning reported).

Observed behavior:
- For 7B: merging improved performance, surpassing both SFT and GRPO checkpoints in accuracy and token economy.
- For 14B: merging was “more of a compromise.”

Infra takeaways:
- Treat merging as a first-class pipeline stage:
  - track which checkpoints were merged
  - store merge configs
  - run post-merge eval suites
- Note that merge behavior can be **scale-dependent**.

Related: [[model-merging]] (if exists), [[fine-tuning]].

---

## Reliability and debugging in RL runs

### “Language mixing” / garbage output incident
Observed during GRPO training:
- Model starts mixing languages mid-CoT.
Correlated signals:
- KL divergence and gradient norms “exploded” at onset (despite grad clipping).
Root cause hypothesis:
- Hard problems where **all rollouts receive reward = 0** → no diverse learning signal → policy destabilizes.
Fix:
- Add more easy/medium problems to stabilize training.
- DAPO-style online filtering/curriculum avoided the issue by dynamically filtering too-easy and too-hard problems.

Infra takeaways:
- Add automated health checks:
  - fraction of all-zero reward prompts per batch
  - KL/grad norm thresholds with early warnings
  - output-language distribution monitors (simple heuristic)
- Maintain “stability buffers” in the sampling curriculum.

---

## Cost engineering and resource planning

Reported cost constraints:
- DeepScaleR-1.5B run: ~$5,000 for ~10% gain (example cited)
- Scaling to 7B/14B would be extremely expensive
- Their final 14B end-to-end (SFT + GRPO) run reportedly cost **< $800**

Infra takeaways:
- Budget-aware strategy:
  - invest in **data quality** (SFT) first
  - apply RL as a targeted second phase
- Cost tracking should be integrated into the experiment system:
  - $/step, $/rollout-token, $/improvement point

Related: [[cost-optimization]].

---

## Reported results snapshot (for infra-driven evaluation)

### 14B (selected metrics)
- At 16K token budget:
  - DeepSeek-R1-Distill-14B: AIME’25 Maj@32 ≈ 0.671
  - Their “Last GRPO 14B”: AIME’25 Maj@32 ≈