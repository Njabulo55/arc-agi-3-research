"""Restart test (Luby/Gomes): is the Duck's time-to-next-level heavy-tailed with a DECREASING hazard?

  python analysis/restart_hazard.py <agent_log_dir> 09:37:30 11:49:32

From analysis-event transcript headers (HH:MM:SS) and levels, build per game the episodes
"time from the previous level-up (or start) to the next level-up"; the final stall is censored at
the run end. Kaplan-Meier -> hazard per time bin. Restarts help when the hazard falls with elapsed
stall time (the Las Vegas speedup condition, Luby et al. 1993).
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HDR = re.compile(r"--- analysis_step=\d+ \| action=\d+ \| (\d\d:\d\d:\d\d)")


def secs(t, t0):
    d = (datetime.strptime(t, "%H:%M:%S") - datetime.strptime(t0, "%H:%M:%S")).total_seconds()
    return d if d >= 0 else d + 86400


def main():
    root, t0, t1 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    end = secs(t1, t0)
    episodes = []           # (duration_s, event_observed)
    for f in sorted(root.glob("*_events.jsonl")):
        pts = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                if e.get("type") != "analysis":
                    continue
                m = HDR.search(e.get("transcript") or "")
                if m:
                    pts.append((secs(m.group(1), t0), int(e.get("level", 1))))
        pts.sort()
        ups, lvl = [], pts[0][1] if pts else 1
        for t, l in pts:
            if l > lvl:
                ups.append(t); lvl = l
        prev = 0.0
        for t in ups:
            episodes.append((t - prev, True)); prev = t
        episodes.append((end - prev, False))
        print(f"{f.name[:4]}: level-ups at {[round(t / 60) for t in ups]} min; final stall {round((end - prev) / 60)} min (censored)")
    # Kaplan-Meier hazard per 15-min bin of stall time
    bins = list(range(0, int(end) + 900, 900))
    print(f"\n{'stall time (min)':>18} {'at risk':>8} {'level-ups':>10} {'hazard/15min':>13}")
    for lo, hi in zip(bins, bins[1:]):
        at_risk = sum(1 for d, _ in episodes if d >= lo)
        ev = sum(1 for d, o in episodes if o and lo <= d < hi)
        if at_risk:
            print(f"{lo // 60:>8}-{hi // 60:<9} {at_risk:8d} {ev:10d} {ev / at_risk:13.2f}")
    obs = sorted(d for d, o in episodes if o)
    print(f"\nobserved level-up waits: n={len(obs)}, median {obs[len(obs) // 2] / 60:.0f} min; "
          f"censored stalls: {sum(1 for _, o in episodes if not o)} (median {sorted(d for d, o in episodes if not o)[len(episodes) - len(obs) - 1 >> 1] / 60:.0f} min)")


if __name__ == "__main__":
    main()
