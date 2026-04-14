---
slug: ml_system_design
sources:
- hav4ik.github.io
tags: []
title: ML System Design
updated: '2026-04-14'
---

# ML System Design

ML system design is the end-to-end engineering discipline of building machine learning systems that are **trainable, evaluable, deployable, and maintainable** under real-world constraints (compute, data availability, latency, budget, safety, and iteration speed). It spans more than “model architecture”: it includes **data pipelines, training/inference infrastructure, evaluation methodology, and operational feedback loops**.

This page focuses especially on lessons from building **long-context reasoning models** (math/olympiad style), where training and evaluation behave differently than short-form tasks.

---

## Scope and core components

A practical ML system design typically includes:

- **Problem definition**
  - Task framing and success metrics (e.g., Pass@1 vs majority-vote accuracy).
  - Constraints: cost, latency, privacy, and allowable data sources.
- **Data system**
  - Data sourcing, cleaning, deduplication, and difficulty/quality filtering.
  - Contamination control and benchmark hygiene.
- **Training system**
  - Choice of training stages (e.g., [[supervised_fine_tuning]] then [[reinforcement_learning]]).
  - Scaling strategy: context length, model size, optimization schedule.
- **Evaluation system**
  - Offline evaluation suite and reproducibility code.
  - Multiple metrics and token-budget-aware testing.
- **Inference system**
  - Sampling strategy, stopping conditions, test-time scaling (e.g., majority voting).
  - Token budgeting and cost/accuracy tradeoffs.
- **Engineering and operations**
  - Framework selection, distributed training, VRAM management, and throughput bottlenecks.
  - Debugging failures (mode collapse, reward collapse, language mixing, instability).

---

## Case study: improving DeepSeek R1 distilled models for math reasoning

A detailed example of system design tradeoffs comes from a team effort to improve **DeepSeek-R1-Distill** math reasoning models (7B and 14B) using **SFT + GRPO** (a lightweight policy-gradient RL variant introduced in DeepSeek Math).

### Reported outcomes (AIME’25)

- **14B model**: **75.8% Maj@32** on AIME’25 (**+8.7%** over baseline).
- **7B model**: **65.8% Maj@32** on AIME’25 (**+7.5%** over baseline).
- Cost claim: a full end-to-end run (SFT + GRPO) for the final 14B model was reported to cost **< $800** (self-funded compute).

These results highlight a common ML system design theme: large improvements can come from **data + training curriculum + evaluation rigor**, not only from architectural changes.

Related concepts/pages to link:
- [[llm_reasoning]]
- [[rlhf]] / [[reinforcement_learning]]
- [[evaluation]]
- [[data_curation]]
- [[distributed_training]]

---

## Training pipeline design (SFT → RL)

A practical pattern used here:

1. **Stage 1: Supervised Fine-Tuning (SFT)**
   - Motivation: full reliance on RL was deemed too costly at larger scales.
   - SFT done on a curated dataset emphasizing **high-quality solution traces** and manageable lengths.
   - Reported setup: trained on an **8×H100** node for 6 epochs at **16K context** (for 7B and 14B bases).

2. **Stage 2: Reinforcement Learning (GRPO)**
   - RL used to further improve reasoning performance and **steer behavior** (notably output length).
   - For 7B: GRPO done in two stages (8K then 16K context).
   - For 14B: RL experiments included 16K context; final submission included a merged approach (see “Model merging” below).

This separation (“skill acquisition” via SFT, “behavior steering” via RL) is a reusable system design pattern when RL is expensive and unstable.

---

## Data system design: curation, difficulty filtering, and trace length

### Goals
- Collect **high-quality** math problems and **correct reasoning traces**, while controlling:
  - dataset difficulty (avoid too-easy items),
  - contamination risk,
  - and reasoning trace length (to manage compute and reduce verbosity).

### Key design choices reported

- **Trace length cap**: focus on solution traces **≤ 16K tokens**
  - Rationale: many correct DeepSeek R1 outputs are reportedly under ~6K tokens; 16K is a tradeoff between accuracy and compute cost.
- **Initial pool**: filtered math word problems from **NuminaMath-1.5** emphasizing:
  - Algebra, Geometry, Number Theory, Combinatorics
  - Sources: Olympiads, AoPS, AMC/AIME, references, etc.
- **Join with correct traces**: used correct R1 reasoning traces from **OpenR1-Math-220k**
  - Result: ~800K → ~27K after filtering/joining.
- **Difficulty filtering via sampling**
  - Sampled multiple solutions per problem (e.g., 8) and removed “easy” problems.
  - Kept problems with **7 or fewer correct solutions** (i.e., not trivially solved by the current model under sampling).
  - Result: ~8K problems retained from one stage of filtering.
- **Add Light-R1 data subset**
  - Added a subset of Light-R1 stage 2 data, deduped and filtered to keep ≤16K CoT.
  - Result: ~2K additional samples after filtering.
- **Avoided data sources**
  - Avoided **cn_k12** (claimed to reduce performance on harder problems).
  - Avoided synthetic math datasets like **Orca-Math** (claimed weaker generators/validators; better for training from scratch than fine-tuning a strong reasoner).

### Resulting dataset usage
- Final SFT dataset: **~10K samples** (≈8K from NuminaMath filtering + ≈2K from Light-R1 subset).
- Harder half used for GRPO.

Design takeaway: for reasoning models, *dataset curation is part of system design*, and **difficulty filtering is effectively curriculum design**.

Related pages:
- [[dataset_curation]]
- [[curriculum_learning]]
- [[data_contamination]]

---

## RL system design: GRPO pitfalls and mitigations

### GRPO vs PPO (resource-driven decision)
- GRPO removes the critic/value model, reducing memory and training time (claimed ~half).
- Tradeoff: potentially less effective than PPO; can have implementation-specific pathologies.

### Length bias (“length hacking”)
Observed issue: with GRPO, models may learn to produce increasingly long solutions.

**Proposed mechanism** (as summarized from DAPO / Dr. GRPO discussions):
- Per-sample loss normalization can cause tokens in longer sequences to contribute less to the loss, implicitly favoring long outputs.
- Fix: **remove per-sample normalization** and compute loss uniformly over tokens.

This is an important ML system design point: *an “algorithm” may behave differently depending on seemingly small implementation details*.

Related pages:
- [[policy_gradient]]
- [[reward_modeling]] (even when using outcome rewards, reward design matters)
- [[training_stability]]

### Difficulty bias (advantage normalization pathology)
- The original GRPO advantage scaling term using inverse reward standard deviation can bias toward questions that are **too easy or too hard**, where outcomes are almost all 1s or 0s.
- Dr. GRPO mitigation described: **remove reward scaling entirely**.

### Do we need KL regularization?
- GRPO often uses **forward KL** regularization to a reference policy, unlike PPO’s reverse KL in classic RLHF formulations.
- Reported mixed findings:
  - Some anecdotal reports of worse results with forward KL.
  - Author’s small-scale experiment reported negligible difference on 1.5B.
  - DAPO argument: remove KL entirely for long-CoT regimes because the model diverges enough that regularization may not help.

**Potential contradiction / open question**
- Some practices emphasize KL as essential for stability in RLHF-like training, while this account suggests **KL may be removable** in long-CoT GRPO settings (at least in some regimes). Treat as **context-dependent**; stability may hinge on other controls (filtering, clipping, reward shaping).

Related pages:
- [[kl_divergence]]
- [[rlhf]]
- [[regularization]]

---

## Inference and evaluation system design

### Token-budget-aware evaluation and “test-time scaling economy”
Evaluation was designed around **sampling many traces** and measuring:
- **Pass@1**
- **Maj@K** (e.g., Maj@32 via majority vote over K samples)
- **Average length**
- performance vs **token budget** (e.g., 8K, 12.8K, 16K, 20K, etc.)

Key insight reported:

- Models can perform **worse** at **higher max token budgets** for Maj@32 (e.g., 32K worse than 16K).
  - Hypothesis: with more “thinking time,” models may **self-doubt** or drift into incorrect answers.
- Conclusion stated explicitly: **better Pass@1 does not imply better Maj@32**.

This impacts system design decisions:
- Always evaluate under the **inference policy you will actually use** (token limits, stop conditions, sampling temperature/top_p).
- Use multiple metrics; for reasoning systems, majority vote accuracy can behave differently than single-sample accuracy.

Related pages:
- [[test_time_scaling]]
- [[majority_voting]]
- [[benchmarking]]

### Contamination control by benchmark choice
- AIME 2025 was treated as “uncontaminated” because it was released after DeepSeek R1 training, and their stated data source (NuminaMath-1.5) was collected beforehand.

Note: “uncontaminated” is an operational claim; true contamination control often requires:
- hashing/dedup against training corpora (when available),
- prompt leakage checks,
- and careful audit trails.

---

## Model merging as a system design tool

The team used **MergeKit** with **TIES** merging:

- Merged SFT and GRPO checkpoints with weights=1 and density=1 (no hyperparameter search reported).

Observed behavior:
- For **7B**, merging improved overall performance and token economy: merged model surpassed both SFT and GRPO checkpoints.
- For **14B**, merging was “more of a compromise.”

Design takeaway: checkpoint merging can be an effective engineering shortcut when:
- you have multiple partially-good behaviors across runs,
- you lack budget for exhaustive RL sweeps,
- but results can be scale-dependent.

Related pages:
- [[model_merging]]
- [[fine_tuning]]

---

## Engineering system design for long-context RL

Long-context RL (8K–24K+ token rollouts) creates distinct engineering constraints compared to common short tasks (GSM8K/MATH with <1K tokens):

### Framework selection and feature requirements
Reported needs for long-context GRPO:
- distributed training (FSDP, offloading),
- sequence packing,
- long-sequence attention optimizations (e.g., Ulysses),
- vLLM rollouts,
- model colocation (“hybrid engine”) to reduce VRAM duplication,
- trainer/rollout integration.

The account contrasts:
- **veRL (HybridFlow)**: described as “battle-tested” for long-context RL (used by DeepScaleR); supports collocation, packing, offloading.
- **TRL/Open-R1**: historically had scaling limitations but reportedly improved over time (added features like Ulysses support and Dr. GRPO techniques).

**Contradiction / staleness warning**
- A table in the source was noted as “outdated”: ecosystem capability changes quickly (e.g., OpenRLHF and TRL added features). System design documentation should version-lock assumptions about framework capabilities.

Related pages:
- [[trl]]
- [[verl]]
- [[vllm]]
- [[fsdp]]

### Training bottlenecks: throughput, not just VRAM
Key bottlenecks described:
- Rollout generation dominates step time.
- “Idle GPU” gaps occur while waiting for the longest sequence in a batch.
- Offload phases add overhead.

Mitigations mentioned:
- increase problems/rollouts per step to reduce relative idle time,
- reuse rollouts via multiple optimization steps (similar to “data echoing”),
- asynchronous RL (not used due to limited nodes).

### LoRA + FSDP + vLLM complexity
While [[lora]] is often considered easy, combining:
- LoRA adapters,
- FSDP sharding,
- a reference model (for KL/reward),
- and vLLM offloading / tensor-parallel rollout engines

creates significant integration risk:
- dtype mismatches with DTensors and gradients,
- VRAM spikes during asynchronous merging and engine startup,
- complex layer wrapping requirements.

Design takeaway: when selecting parameter-efficient fine-tuning methods, incorporate **integration complexity** into the system design cost model, not only VRAM savings.

Related pages:
- [[lora]]
- [[parameter_efficient_fine_tuning]]
- [[distributed_systems]]

---

## Failure modes and debugging patterns

### “Language mixing” during RL
Observed during 14B GRPO runs (notably in Open-R1/TRL runs):
- sudden insertion of other languages mid chain-of-thought,
- exploding KL divergence and gradient norms around onset.

Root cause hypothesis:
- very hard prompts where **all rollouts receive reward 0** → no learning signal diversity → model spirals into garbage behavior.

Mitigations:
- add more easy/medium problems to stabilize.
- use DAPO-like online filtering that removes too-easy and too-hard samples, acting as an **online adaptive curriculum**.

This is a general ML system design lesson: RL pipelines need **guardrails against reward collapse** and **batch degeneracy**.

Related pages:
- [[reward_shaping]]
- [[curriculum_learning]]
- [[training_debugging]]

---

## Design heuristics extracted from the case study

- **Treat token budget as a first-class resource**
  - Optimize “accuracy per token,” not only raw accuracy.
- **Separate capability learning from behavior shaping**
  - Use SFT for acquiring core competence; use RL for steering (length, style, stability).
- **Curriculum matters**
  - Difficulty filtering (offline and online) can stabilize RL and improve results.
- **Beware metric mismatch**
  - Pass@1 improvements may not translate to Maj@K; larger token budgets can reduce Maj@K.
- **Implementation details can dominate algorithmic intent**
  - GRPO length bias can be “just” a normalization/aggregation detail.
- **Framework capability is part of the model**
  - Long-context RL success depends heavily on distributed systems features (packing, offload, collocation, rollout engines).
- **Scaling tricks may not transfer**
  - Iterative context lengthening reportedly helped at 1.5B but harmed 7B/14B long-context performance.

---

## Open questions and unresolved contradictions

- **KL regularization necessity**
  - Some approaches claim KL can be removed for long-CoT; classic RLHF often treats KL as essential. Likely depends on reward design, online filtering, and training regime.
- **LoRA vs full fine-tuning**
  - LoRA converged faster in some runs, but the best reported model used full fine-tuning; without ablations it’s unclear which is better.
- **“More thinking time” can hurt**
  - Contrary to naive expectations that longer rollouts always help, higher max_len budgets sometimes reduced Maj@32. This suggests a need for:
    - better stopping criteria,
    - self-consistency strategies,
    - or inference-time constraints to prevent drift.

---

## Related pages (suggested cross-references)

- [[data_curation]]
- [[supervised_fine_tuning]]
- [[reinforcement_learning]]
- [[rlhf]]
- [[evaluation]]
- [[test_time_scaling]]
- [[majority_voting]]
- [[model_merging]]
- [[lora]]
- [[fsdp]]
- [[vllm]]
- [[trl]] / [[verl]]
- [[training_stability]]