"""C3 (pre-registered): Gittins-index scheduling vs processor sharing, simulated from measured hazards.

  python analysis/gittins_c3.py

Solve-time distribution: Kaplan-Meier from R0 (Duck run, anim-a): 39 observed level-up waits and
25 censored stalls (minutes, under processor sharing). Mass beyond the last observation = never.
Model: 25 games, sequential levels, i.i.d. level solve times; 25 slot-minutes per minute; a game
may use <= 10 slots per minute (its ~90% queue wait caps its speed-up); horizon 132 min.
Score = sum of level indices solved (later levels weigh more, as in the benchmark).
"""
import numpy as np

OBS = []  # filled from restart_hazard-style episodes below
# Episodes measured in R0 (minutes): observed level-up waits and censored final stalls.
OBSERVED = [7, 31, 63, 14, 79, 16, 16, 39, 37, 56, 18, 2, 21, 25, 23, 40, 24, 72, 31, 11, 21, 13, 12, 17, 53,
            40, 25, 3, 48, 28, 22, 16, 48, 28, 16, 20, 15, 12, 38]
CENSORED = [94, 132, 69, 118, 53, 24, 132, 76, 114, 21, 108, 29, 100, 37, 67, 129, 132, 132, 84, 104, 110, 40,
            96, 105, 94]
H, GAMES, SLOTS, CAP = 132, 25, 25, 10


def km_pmf():
    """Discrete per-minute pmf of solve time (index t = solved during minute t+1); rest = never."""
    times = sorted(set(OBSERVED))
    surv, pmf = 1.0, np.zeros(H + 1)
    for t in times:
        at_risk = sum(1 for d in OBSERVED if d >= t) + sum(1 for d in CENSORED if d >= t)
        d = sum(1 for x in OBSERVED if x == t)
        p = surv * d / at_risk
        pmf[min(t, H)] += p
        surv -= p
    return pmf, surv


PMF, NEVER = km_pmf()
SURV = 1.0 - np.concatenate([[0.0], np.cumsum(PMF)])        # SURV[a] = P(T > a)


def gittins(age, weight):
    """max over horizon d of weight*P(solve in (age, age+d] | T > age) / E[time used in window | T > age]."""
    s0 = SURV[min(age, H)]
    if s0 <= 1e-9:
        return 0.0
    best, p_cum, t_cum = 0.0, 0.0, 0.0
    for d in range(1, H - age + 1):
        a = age + d
        t_cum += SURV[min(a - 1, H)] / s0                  # expected minutes used in step d
        p_cum += PMF[min(a, H)] / s0
        best = max(best, weight * p_cum / t_cum)
    return best


GIT = {}


def sample(rng):
    u = rng.random()
    c = np.cumsum(PMF)
    i = int(np.searchsorted(c, u))
    return i if i <= H and u <= c[-1] else 10 ** 9


HETERO = False       # True: a level is "impossible" with P(never) and restarts cannot change that


def draw_level(rng):
    """-> (impossible_flag, need). Heterogeneous model fixes the level's type at level start."""
    if HETERO:
        imp = rng.random() < NEVER
        if imp:
            return True, 10 ** 9
        while True:
            n = sample(rng)
            if n < 10 ** 9:
                return False, n
    return False, sample(rng)


def resample(rng, imp):
    if HETERO:
        if imp:
            return 10 ** 9
        while True:
            n = sample(rng)
            if n < 10 ** 9:
                return n
    return sample(rng)


def run(policy, restart, rng):
    types = [draw_level(rng) for _ in range(GAMES)]
    imp = [t[0] for t in types]
    need = [t[1] for t in types]                            # work needed for current level
    age = [0] * GAMES                                       # work spent on current level
    level = [1] * GAMES
    score = 0
    for minute in range(H):
        if policy == "ps":
            alloc = [1] * GAMES
        else:
            idx = []
            for g in range(GAMES):
                key = (min(age[g], H), level[g])
                if key not in GIT:
                    GIT[key] = gittins(min(age[g], H), level[g])
                idx.append(GIT[key])
            alloc, left = [0] * GAMES, SLOTS
            for g in sorted(range(GAMES), key=lambda g: -idx[g]):
                take = min(CAP, left)
                alloc[g] = take; left -= take
                if left == 0:
                    break
        for g in range(GAMES):
            for _ in range(alloc[g]):
                age[g] += 1
                if age[g] >= need[g]:
                    score += level[g]; level[g] += 1
                    imp[g], need[g] = draw_level(rng); age[g] = 0
                elif restart and age[g] >= restart:
                    need[g], age[g] = resample(rng, imp[g]), 0   # fresh draw of the SAME level
    return score


def main():
    global HETERO
    import sys
    HETERO = "--hetero" in sys.argv
    print("MODEL:", "heterogeneous (impossible levels stay impossible)" if HETERO else "i.i.d. restarts (optimistic)")
    rng = np.random.default_rng(0)
    print(f"KM: P(solve within 132 min) = {1 - NEVER:.2f}; median observed wait {np.median(OBSERVED):.0f} min")
    res = {}
    for name, pol, rs in [("PS (today)", "ps", 0), ("PS + restart@40", "ps", 40), ("Gittins", "git", 0), ("Gittins + restart@40", "git", 40)]:
        sc = [run(pol, rs, rng) for _ in range(400)]
        res[name] = np.mean(sc)
        print(f"{name:22} weighted levels {np.mean(sc):7.1f} ± {np.std(sc) / np.sqrt(len(sc)):.1f}")
    base = res["PS (today)"]
    best = max((v, k) for k, v in res.items() if k != "PS (today)")
    print(f"\nbest non-PS: {best[1]} = {100 * (best[0] / base - 1):+.0f}% vs PS  ->  "
          f"{'PASS (build it)' if best[0] >= 1.15 * base else 'FAIL (do not build)'}")


if __name__ == "__main__":
    main()
