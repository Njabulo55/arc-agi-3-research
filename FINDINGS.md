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

Status: GPU A/B tests of 7 and 9 on the real agent are scheduled.
