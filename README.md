# ARC-AGI-3 Research Notes & Final Competition Retrospective

Independent research on the [ARC Prize 2026](https://arcprize.org) (ARC-AGI-3) by Njabulo Mthethwa, student at the University of Cape Town. 

In ARC-AGI-3, an autonomous agent must discover the latent mechanics of novel, interactive 2D grid-world environments and solve multi-level combinatorial puzzles from scratch—with zero human instructions or prior environment specs—under strict execution time and compute constraints.

---

## Research Methodology & Core Operating Principle

**Pre-registration and complete transparency:** Every experiment in this project received a written hypothesis, an explicit quantitative success threshold, and a pre-registered kill rule *before* a single GPU cycle was spent. Every result—especially negative results and failed directions—is logged with its exact mechanism check in [FINDINGS.md](FINDINGS.md).

---

## Executive Summary & Final Competition Breakthrough

Over the course of the competition, our research progressed from diagnosing why standard LLM-in-a-loop agents stall on ARC to building a high-throughput, dual-system cognitive architecture:

1. **Throughput & Prefix Cache Bottleneck Unlocked:** Standard inference engines suffer massive context thrashing under multi-turn game rollouts. By deploying a W4A16 AutoRound quantized target model (`Qwen3.8-Flash-Next`) paired with an Albucino NEXTN speculative drafter on SGLang (`MEMFRAC=0.96`, `MAMBA_CACHE=60`), we achieved **619.32 aggregate generated tokens/sec** across 10 concurrent game streams.
2. **The Fast-Slow Dual System:** 
   - **System 1 (Fast Symbolic Macro-Planner):** Tracks board state transitions, induces discrete Moore-machine state graphs, computes reachability, and synthesizes deterministic multi-step navigation macros. This offloaded >70% of low-level exploration steps.
   - **System 2 (Slow Prefix-Frozen LLM Reasoner):** Leverages high-throughput speculative decoding strictly for high-level hypothesis formulation and sub-goal planning, maintaining strict append-only history buffers to preserve a 78.4% Radix prefix cache hit rate.
3. **Benchmark Breakthrough on `hard10`:**
   - On the standardized 10-game competition benchmark (`hard10`), our final architecture (**Fast-Slow v9**) achieved a **27.91 mean score** (+18.2% relative gain) and a **24.61 median score** (+64.1% relative gain), clearing **32 / 74 total levels** in 48m 32s (1.80M tokens generated).
   - This directly surpassed the previous top reference baseline (`arc3-ctl-h10`, which scored 30.58 on the Kaggle public leaderboard and had an offline `hard10` mean of 23.61 / median of 15.00 with 30 levels).
   - On spatially complex coordination games (`dc22`, `sp80`), Fast-Slow doubled the levels solved over the baseline.

---

## Quantitative Results Comparison (`hard10` Benchmark)

All evaluations scored in competition mode on 10 standardized hard ARC-AGI-3 games (`cn04`, `dc22`, `g50t`, `lf52`, `ls20`, `m0r0`, `s5i5`, `sp80`, `su15`, `wa30`) using NVIDIA RTX 6000 Ada / RTX Pro 6000 hardware:

| Architecture / Arm | Mean Score | Median Score | Levels Solved | Generated Tokens | Throughput (tok/s) | Key Bottleneck / Outcome |
|---|---|---|---|---|---|---|
| **Public Baseline (Qwen REPL)** | ~7.25 | ~0.00 | ~7 / 74 | ~170k | ~55 tok/s | KV cache trashing, severe queue stalls, raw primitive actions |
| **SGLang Speculative Draft (M4)** | 0.24 | 0.00 | 2 / 74 | 141k | 117.6 tok/s | Speculative draft validated, but high concurrency lock contention |
| **Fast-Slow v5 (Dynamic WM)** | 5.75 | 5.45 | 12 / 74 | 658k | 212.8 tok/s | Dynamic in-context prompt edits destroyed Radix prefix cache |
| **Fast-Slow v7 (Prefix Repaired)** | 13.81 | 11.20 | 23 / 74 | 1.15M | 395.0 tok/s | Recovered prefix caching; throttled by unoptimized safe-tensors load |
| **Reference Baseline (`arc3-ctl-h10`)** | 23.61 | 15.00 | 30 / 74 | 1.77M | 608.3 tok/s | Public Leaderboard score: **30.58**; strong throughput, primitive macros |
| **Fast-Slow v9 (Final Architecture)** | **27.91** | **24.61** | **32 / 74** | **1.80M** | **619.3 tok/s** | **+18.2% Mean, +64.1% Median**; 10 parallel games, 0 OOMs, 48m 32s |

---

## Deep Dive: The Test-Time Computing & Terminal-RSI Post-Mortem

A major hypothesis investigated during this competition was whether **Test-Time Training (TTT)** and online **Recursive Self-Improvement (RSI)** could allow an agent to adapt to novel game mechanics on the fly. We adapted the `Terminal-RSI` framework (originally built for bash/terminal self-correction) to ARC-AGI-3, executing online Token-LoRA rollouts with hindsight experience replay (RSR) and policy-critic updates across 6 rounds on difficult puzzle games (`ar25`, `bp35`, `sk48`):

### What the Experiments Revealed:
1. **End-to-End Execution & Signs of Life:** The online adaptation engine successfully ran end-to-end for 117 minutes, training 3 consecutive LoRA policy versions (`v0`, `v1`, `v2`). On `ar25-0c556536`, score temporarily climbed to 16.67 in Round 4 (solving 2 levels).
2. **Policy Drift & The Sparse Reward Wall:** By Rounds 5 and 6, the agent degraded back to 0.93. In discrete combinatorial environments where rewards are sparse (only awarded upon completing an entire level), continuous gradient backpropagation on a handful of rollouts causes severe credit assignment failure and catastrophic forgetting.
3. **The Throughput Penalty:** Performing online backpropagation and swapping LoRA adapters into the inference engine continually invalidated the Radix KV cache, dropping generation throughput from 619.3 tok/s down to **80.03 tok/s** (a 7.7× speed penalty).
4. **Core Scientific Conclusion:** In discrete, combinatorial problem domains like ARC, test-time compute is drastically more effective when allocated to **symbolic program synthesis, state-machine induction, and macro-search** rather than online parametric gradient adaptation.

---

## Compute Stewardship & Credit Discipline

Access to high-end compute (including GPU grants and cloud credits) is a privilege that demands rigorous stewardship. In this project:
- Compute spending on external cloud platforms (e.g., Vast.ai) was kept strictly minimal.
- When an exploratory direction (such as online LoRA backprop) showed diminishing returns and failed pre-registered bars, the arm was immediately halted and archived rather than burning credits on brute-force scaling.
- By optimizing concurrency, Mamba cache memory allocation (`MEMFRAC=0.96`), and FUSE precaching, the full 10-game `hard10` suite was successfully evaluated within standard single-GPU envelopes without wasting resources.

---

## Research Horizon: From Discrete Puzzles to Biomolecular Modeling

The lessons learned from ARC-AGI-3 extend directly to complex physical and biological systems. Discovering latent transition rules in grid worlds is fundamentally homologous to navigating conformational energy landscapes in structural biology. 

Specifically, **RNA 3D tertiary structure prediction**—understanding how single-stranded RNA molecules fold into complex pseudoknots, tertiary motifs, and functional riboswitches—presents identical algorithmic challenges:
- Immense combinatorial search spaces.
- Sparse non-local energetic stabilization (analogous to goal detection).
- The necessity of hybrid search: fast deterministic biophysical constraints (System 1) guiding generative macro-sampling (System 2).

We intend to carry the empirical rigor, prefix-cached high-throughput inference mechanisms, and symbolic-neural hybrid architectures developed here into open-source research on biomolecular structure prediction.

---

## Repository Structure

```
├── README.md               # Project overview and final competition retrospective
├── FINDINGS.md             # Complete chronological log of 15 pre-registered hypotheses and outcomes
├── TECHNICAL_REPORT.md     # In-depth architectural breakdown, serving specs, and mathematical formulation
├── tools/
│   ├── competition_mode.py # Strict emulation of Kaggle competition gateway scoring
│   ├── evaluate.py         # Local multi-game evaluator and scorecard logger
├── analysis/
│   ├── prefix_reuse_sim.py # Radix prefix cache reuse simulation across context policies
│   ├── restart_hazard.py   # Kaplan-Meier survival hazard analysis of level completion times
│   └── gittins_c3.py       # Multi-armed bandit Gittins-index scheduling simulation
└── notebooks/
    └── arc3_fast_slow_submission.ipynb # Complete standalone competition submission notebook
```

---

## License

This project is licensed under the [MIT License](LICENSE).
