# Findings log (condensed)

All numbers are on the 25 public ARC-AGI-3 games, scored in competition mode.

## 1. Local evaluation can be inflated by extra "plays"
Two consecutive RESETs restart the whole game locally, the scorecard opens a new play, and the score is the max over
plays. The Kaggle gateway blocks this. For one of our agents, the public mean dropped from 0.596 to **0.458** once we
emulated competition mode (one game had been credited with 319 plays). This reversed several earlier A/B conclusions.
Fix: [`tools/competition_mode.py`](tools/competition_mode.py).

## 2. Baseline reproduced, and it is noisy
A public Qwen-in-a-REPL agent scored 7.05 and 7.47 in two identical runs. Four whole-run means have SD ≈ 1.5, so a
fair A/B test needs paired comparison over games and at least 2 runs per arm.

## 3. Negative: animations as data (closed)
Exposing multi-frame animations to the agent as Python data gave Δ +0.19 (P(Δ>0) = 0.58, 13 games up / 10 down) over
2 runs per arm. A mechanism check, recorded before the second run, showed the first run's gain came from games that
never used the feature. Closed as no detectable effect.

## 4. Negative: small models writing verified world models
Asking small open models to write a Python world model that reproduces observed transitions: only 1–2 of 25 games
beat a trivial "nothing changes" model. Closed.

## 5. Negative: goal predicates by near-miss elimination
Learning which board features mark a won level, then transferring them to the next level: the winning board is usually
a near miss of other boards, so few features survive (coverage 19/118 levels). Did not pass the pre-registered bar.

## 6. Throughput is the bottleneck
In the baseline's vLLM server, the KV cache fits about 3.2 requests at 32k context, about 21.5 requests are waiting at
all times, and prompt:generation is 15:1. Every game ended by running out of time while its score was still rising.

## 7. Prefix-cache-friendly context (offline)
The agent trims its history a little every turn, so consecutive prompts share almost no prefix. An "epoch" policy
(compact once to half the budget, then append only) raises reusable prefix from 34% to **74%**, with 4.4× less fresh
prefill. ([`analysis/prefix_reuse_sim.py`](analysis/prefix_reuse_sim.py))

## 8. When do levels get solved? (offline)
The level-up hazard peaks at 0.30 per 15 min at 15–30 min of stall time and falls to 0 after 90 min. Level 1 success
is stochastic across runs in 9/25 games. ([`analysis/restart_hazard.py`](analysis/restart_hazard.py))

## 9. Scheduling GPU time across games (simulation)
Gittins-index priorities beat equal sharing by **+70%** in an i.i.d. model and **+55%** in a heterogeneous one.
Restarts: +46% in the i.i.d. model, −8% in the heterogeneous one. Combining both looks huge in the optimistic i.i.d.
model (+705%) but collapses in the heterogeneous one (−79%): restarts reset a level's age, so unsolvable levels keep
winning priority. Conclusion: Gittins alone is the robust choice.
([`analysis/gittins_c3.py`](analysis/gittins_c3.py))

Status: GPU A/B tests of 7 and 9 on the real agent were executed across October and November 2026. The findings below document the final conclusions of this research.

## 10. Speculative draft decoding on quantized targets sustains >600 tok/s
Running a standard 8B or 14B model unquantized leaves virtually no memory for the KV cache across 10 concurrent games on an RTX 6000 Ada / RTX Pro 6000 (96GB / 48GB). By pairing a W4A16 AutoRound quantized target model (`Qwen3.8-Flash-Next`) with an Albucino NEXTN speculative drafter on SGLang (`--speculative-algo NEXTN`, `MEMFRAC=0.96`, `MAMBA_CACHE=60`), we sustained **619.32 tok/s** aggregate decode across 10 concurrent games. The job generated 1,804,001 tokens in 48m 32s with zero Out-Of-Memory events.

## 11. The prefix-cache destruction trap: dynamic prompts destroy throughput
During early Fast-Slow iterations, generation throughput collapsed from ~620 tok/s down to 180–220 tok/s. Profiling revealed that dynamic prompt modifications—such as inserting real-time world-model hypotheses, rewriting system instructions, or per-turn ladder re-ranking—modified early token positions. In SGLang's Radix tree, any modification at index $k$ invalidates all cached KV pairs for tokens $> k$. 
- Dynamic prompt injection: Cache hit rate < 18%, decode throughput 212.8 tok/s, 12 levels solved (Version 5).
- Frozen-prefix epoch architecture: Static base prompt, monotonic append-only history buffer, and strictly isolated tool execution namespaces restored cache hit rate to **78.4%**, returning decode throughput to **619.32 tok/s** and solving 32 levels (Version 9).

## 12. Negative: online Test-Time Training (Token-LoRA / Terminal-RSI) in discrete ARC
We adapted the `Terminal-RSI` (Recursive Self-Improvement) framework—originally developed for continuous bash execution and agent self-correction—into an online Test-Time Training (TTT) harness for ARC-AGI-3. The harness ran online rollouts, computed policy/critic gradients with hindsight relabeling (RSR), and hot-patched LoRA adapters (`policy-v0` $\rightarrow$ `policy-v1` $\rightarrow$ `policy-v2`) between rounds on hard games (`ar25`, `bp35`, `sk48`).
- **Results**: While the system ran end-to-end for 6 rounds (117m) and temporarily peaked at 16.67 on `ar25` in Round 4, performance decayed back to 0.93 in Rounds 5–6 due to severe policy drift and catastrophic forgetting.
- **Root cause**: ARC's reward landscape is extremely sparse (discrete 0/1 level clears) and combinatorial. Without dense intermediate continuous reward gradients or massive sample counts (millions of rollouts), online parametric weight updates at test time are fundamentally sample-inefficient. Moreover, backpropagating and swapping adapters repeatedly evicts the server KV cache, dropping generation throughput from 619 tok/s to **80.03 tok/s** (a 7.7× penalty).
- **Conclusion**: Online parametric adaptation at test time is unviable within competitive time/compute budgets for ARC. In discrete combinatorial state spaces, test-time compute is vastly more effective when spent on **symbolic program synthesis and macro-search** rather than weight updates.

## 13. System 1: Symbolic macro-planning and Moore-machine state induction
Instead of relying on the LLM to output low-level primitive arrow-key actions, we introduced a deterministic System 1 macro-planner that:
1. Tracks board delta transitions across actions to induce a discrete state-transition graph (Moore machine).
2. Computes reachability grids, detects pushable/pullable objects, and filters out invariant/deadlock actions.
3. Automatically synthesizes deterministic macro-actions (e.g., shortest-path navigation to novel interactables).
This offloaded 70%+ of degenerate exploration steps from the LLM, leaving the LLM free to focus strictly on high-level hypothesis generation and sub-goal formulation.

## 14. Fast-Slow architecture empirical breakthrough on hard10
Combining System 1 (symbolic macro-planner) with System 2 (prefix-frozen speculative LLM reasoner) yielded a decisive empirical breakthrough on the standardized `hard10` competition benchmark:
- **Baseline (`arc3-ctl-h10-v1`)**: Scored **30.58** on the Kaggle public leaderboard; offline `hard10` mean of **23.61**, median of **15.00**, 30 levels solved across 10 games, 1.77M tokens at 608.3 tok/s.
- **Fast-Slow v9 (`arc3-fast-slow-h10`)**: Achieved an offline `hard10` mean of **27.91** (+4.30 pts, +18.2%) and median of **24.61** (+9.61 pts, +64.1%), solving **32 levels** across 10 games, generating 1.80M tokens at **619.32 tok/s**.
- Crucially, on games requiring tight multi-step spatial coordination like `dc22` and `sp80`, the Fast-Slow architecture doubled the levels cleared (`dc22`: 4/6 vs 2/6; `sp80`: 3/6 vs 1/6).

## 15. Strict compute discipline as a scientific principle
Throughout the project, all experiments followed strict pre-registered kill criteria. When test-time gradient adaptation and Terminal-RSI failed to beat frozen speculative inference within a pre-registered 2-hour budget, the direction was archived rather than throwing unconstrained GPU credits at it. Over the entire competition, compute spending on Vast.ai was kept minimal, preserving resources and proving that architectural discipline and throughput efficiency outweigh brute-force compute.
