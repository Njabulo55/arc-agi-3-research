# ARC-AGI-3 research notes

Independent research on the [ARC Prize 2026](https://arcprize.org) (ARC-AGI-3) by Njabulo Mthethwa, a student at the
University of Cape Town. In ARC-AGI-3, an agent has to learn unfamiliar interactive puzzle games from scratch, with no
instructions, as efficiently as a person would.

**How I work:** every idea gets a written hypothesis, a success threshold and a kill rule *before* it is run. Every
result is logged, including the ones that didn't work. Negative results are in [FINDINGS.md](FINDINGS.md) too.

## What's here
| Path | What it does |
|---|---|
| [`tools/competition_mode.py`](tools/competition_mode.py) | Makes local scoring match Kaggle's. Without it, local scores can be badly inflated (one of our agents was credited with 319 "plays" on one game). |
| [`tools/evaluate.py`](tools/evaluate.py) | Local evaluator on the public games in competition mode. Reports plays per game. Expects the `arc_agi` package and an agent file. |
| [`analysis/prefix_reuse_sim.py`](analysis/prefix_reuse_sim.py) | Measures how much of an LLM agent's prompts a prefix cache could reuse under different context policies, from its logs. |
| [`analysis/restart_hazard.py`](analysis/restart_hazard.py) | Kaplan–Meier hazard of "time to next level" from agent logs: do restarts help (Luby et al. 1993)? |
| [`analysis/gittins_c3.py`](analysis/gittins_c3.py) | Simulates Gittins-index scheduling of GPU time across games against processor sharing, using the measured hazard. |

## Status (Sept 2026)
- Reproduced the strongest public open-source baseline (a Qwen agent in a Python REPL, ~7% on the public games).
- Found that it is limited by **throughput**, not reasoning. The KV cache fits about 3 concurrent requests, about 21
  requests are always waiting, and prompts are about 15× larger than generations.
- Built three fixes and tested them offline: epoch-style context (prefix caching can hit), Gittins-index request
  priorities, and reasoning restarts. GPU A/B tests are planned. Code for these will be published with the results.

## Next
RNA 3D structure prediction, with the same pre-registered, open approach.

License: MIT.
