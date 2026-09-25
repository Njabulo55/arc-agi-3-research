"""Local evaluation of agent/my_agent.py on the public games, using the official scorecard.

  python tools/evaluate.py [--games ls20,ft09] [--max-actions 4000] [--out results.json]

Each game runs in its own process (the engine and agent are single-threaded).
Reports per game: levels completed / total, actions, official score; and the mean score.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))


def competition_mode(env):
    """Mirror the gateway's competition mode (arc_agi/api.py cmd()): a RESET sent while the game's
    action count is 0 (it would restart the whole game = a new scored play) is NOT executed; the
    current observation is returned and the scorecard records a non-full reset. Without this, local
    runs get extra plays and the scorecard keeps the best one, which Kaggle never allows."""
    from arcengine import GameAction
    orig = env.step

    def step(action, data=None, reasoning=None):
        g = getattr(env, "_game", None)
        if action == GameAction.RESET and g is not None and getattr(g, "_action_count", 1) == 0:
            resp = env.observation_space
            if resp is not None and getattr(env, "scorecard_manager", None) and resp.guid:
                env.scorecard_manager.update_scorecard(resp.guid, resp, False)
            return resp
        return orig(action, data, reasoning)
    env.step = step
    return env


def run_game(args):
    game_id, max_actions, agent_path, env_dir, comp = args
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arc_agi import OperationMode
    spec = importlib.util.spec_from_file_location("user_agent_module", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Agent = mod.MyAgent
    Agent.MAX_ACTIONS = max_actions
    if env_dir:
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    else:
        arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id)
    if comp:
        competition_mode(env)
    t = time.time()
    agent = Agent(card_id="local-eval", game_id=game_id, agent_name=f"eval.{game_id}",
                  ROOT_URL="http://localhost", record=False, arc_env=env, tags=["eval"])
    agent.main()
    sc = arc.get_scorecard()
    d = json.loads(sc.model_dump_json()) if hasattr(sc, "model_dump_json") else {}
    env_scores = []
    for e in d.get("environments", []) or []:
        for run in e.get("runs", []):
            env_scores.append(run)
    best = max(env_scores, key=lambda r: r.get("score", 0)) if env_scores else {}
    final = agent.frames[-1]
    return {"game": game_id, "levels": final.levels_completed, "win_levels": final.win_levels,
            "actions": agent.action_counter, "score": best.get("score", 0.0),
            "consults": getattr(agent, "consults", 0),
            "level_actions": best.get("level_actions"), "level_scores": best.get("level_scores"),
            "baseline": best.get("level_baseline_actions"), "plays": len(env_scores),
            "secs": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=None)
    ap.add_argument("--max-actions", type=int, default=4000)
    ap.add_argument("--agent", default=str(ROOT / "agent" / "my_agent.py"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-dir", default=None, help="offline mode: directory of game folders")
    ap.add_argument("--no-competition", action="store_true",
                    help="allow full game resets (extra scored plays); NOT what Kaggle does")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    env_root = Path(args.env_dir) if args.env_dir else ROOT / "environment_files"
    games = sorted(p.name for p in env_root.iterdir() if p.is_dir())
    if args.games:
        games = [g for g in games if g in args.games.split(",")]
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run_game, [(g, args.max_actions, args.agent, args.env_dir, not args.no_competition) for g in games]))
    for r in res:
        print(f"{r['game']:6} levels {r['levels']:2}/{r['win_levels']:<2} actions {r['actions']:5} "
              f"score {r['score']:6.2f}  plays {r['plays']}  lvl_actions {r['level_actions']}  base {r['baseline']}  ({r['secs']}s)")
    mean = sum(r["score"] for r in res) / max(1, len(res))
    print(f"\nMEAN SCORE {mean:.3f} over {len(res)} games; total levels {sum(r['levels'] for r in res)}; "
          f"{time.time() - t:.0f}s")
    if args.out:
        Path(args.out).write_text(json.dumps({"mean": mean, "games": res}, indent=1))


if __name__ == "__main__":
    main()
