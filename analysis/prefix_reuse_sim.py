"""C1 (research, CPU): how much of each Duck request could prefix caching reuse, under the current
context policy vs an epoch (append-only, compact every K turns) policy?

  python analysis/prefix_reuse_sim.py <agent_log_dir>

Per game, each analysis step's transcript gives the turn's user prompt, then the model/tool blocks
of each request within the turn. Tokens ~ chars / 3.5; an image part ~ 256 tokens.
Duck policy (tool_agent.py): messages = [system] + history + [user(+image)] + in-turn blocks;
history = previous turns, dropped from the FRONT when the request exceeds the context budget
(32,768 - 512 reserve). Epoch policy: history grows append-only; when over budget, compact ONCE
down to BUDGET/2 of most recent turns (a single prefix break), then append again.
Reuse of request r = longest common prefix with request r-1 of the same game / prompt length.
"""
import json
import re
import sys
from pathlib import Path

BUDGET = 32768 - 512 - 6000      # context budget minus reply reserve (max output ~6k)
IMG = 256
SEC = re.compile(r"^\[([A-Z][A-Z :_a-z]*)\]\s*$", re.M)


def tok(s):
    return len(s) / 3.5


def turns_of(path):
    """-> system_tokens, list of turns; turn = (user_tokens, [block tokens per request...])."""
    turns, system, seen = [], None, set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            if e.get("type") != "analysis":
                continue
            t = e.get("transcript") or ""
            if t in seen:
                continue
            seen.add(t)
            parts = SEC.split(t)
            secs = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
            user, blocks, cur = 0.0, [], 0.0
            for name, body in secs:
                if name.startswith("SYSTEM PROMPT"):
                    system = tok(body)
                elif name.startswith("USER PROMPT"):
                    if user == 0.0:
                        user = tok(body) + IMG
                    else:
                        cur += tok(body)           # follow-up nudges inside a turn
                elif name.startswith(("THINKING", "ASSISTANT", "TOOL CALL")):
                    cur += tok(body)
                elif name.startswith("TOOL RESULT"):
                    cur += tok(body)
                    blocks.append(cur); cur = 0.0  # one request ends with its tool result
            if cur:
                blocks.append(cur)
            turns.append((user, blocks or [0.0]))
    return system or 3000.0, turns


def simulate(system, turns, policy, K=None):
    """Return list of (prompt_tokens, reused_tokens) per request."""
    hist = []                                     # list of (turn_tokens) kept in context, oldest first
    out, prev_seq = [], None
    for user, blocks in turns:
        seq_hist = list(hist)
        within = []
        for b in blocks:
            prompt = system + sum(v for _, v in seq_hist) + user + sum(within)
            if policy == "duck":
                while seq_hist and prompt > BUDGET:
                    seq_hist.pop(0)
                    prompt = system + sum(v for _, v in seq_hist) + user + sum(within)
            seq = ("S",) + tuple(("H", id_, v) for id_, v in seq_hist) + (("U", len(out) if not within else "same"),)
            # longest common prefix in tokens with the previous request (same game)
            reused = 0.0
            if prev_seq is not None:
                reused = system
                for a, b_ in zip(prev_seq[0][1:], seq[1:]):
                    if a == b_ and a[0] == "H":
                        reused += a[2]
                    else:
                        break
                if within:                        # same turn: previous request is an exact prefix
                    reused = prev_seq[1]
            out.append((prompt, min(reused, prompt)))
            prev_seq = (seq, prompt)
            within.append(b)
        turn_total = user + sum(blocks)
        hist = seq_hist + [(len(out), turn_total)] if policy == "duck" else hist + [(len(out), turn_total)]
        if policy == "epoch" and system + sum(v for _, v in hist) > BUDGET:
            keep, tot = [], 0.0
            for h in reversed(hist):              # compact once: keep the most recent half-budget
                if tot + h[1] > BUDGET / 2:
                    break
                keep.insert(0, h); tot += h[1]
            hist = keep
    return out


def main():
    root = Path(sys.argv[1])
    agg = {"duck": [0.0, 0.0], "epoch": [0.0, 0.0]}
    for f in sorted(root.glob("*_events.jsonl")):
        system, turns = turns_of(f)
        for pol in agg:
            # history entries carry ids so identical blocks are matched only if they are the same turn
            res = simulate(system, [(u, b) for u, b in turns], pol)
            agg[pol][0] += sum(p for p, _ in res); agg[pol][1] += sum(r for _, r in res)
    for pol, (p, r) in agg.items():
        print(f"{pol:6}: prompt tokens {p / 1e6:6.2f} M | reusable by prefix cache {r / 1e6:6.2f} M ({100 * r / p:.1f}%) | "
              f"fresh prefill {(p - r) / 1e6:6.2f} M")
    d, e = agg["duck"], agg["epoch"]
    print(f"\nfresh-prefill reduction, epoch vs duck-with-cache: {(d[0] - d[1]) / max(1, e[0] - e[1]):.1f}x ; "
          f"vs no cache (today): {d[0] / max(1, e[0] - e[1]):.1f}x")


if __name__ == "__main__":
    main()
