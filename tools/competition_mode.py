"""Emulate the Kaggle gateway's competition mode in local ARC-AGI-3 runs.

In local/offline `arc_agi`, a RESET sent when no action has been taken restarts the whole game and the
scorecard opens a new play; an environment's score is the max over plays, so local scores can be
inflated. The Kaggle gateway (arc_agi/api.py, cmd()) does not execute such a RESET. Wrap your env:

    env = competition_mode(arc.make(game_id))

Then check that your scorecard reports exactly one play per game.
"""
from arcengine import GameAction


def competition_mode(env):
    orig = env.step

    def step(action, data=None, reasoning=None):
        g = getattr(env, "_game", None)
        if action == GameAction.RESET and g is not None and getattr(g, "_action_count", 1) == 0:
            resp = env.observation_space   # not executed, like the gateway
            if resp is not None and getattr(env, "scorecard_manager", None) and resp.guid:
                env.scorecard_manager.update_scorecard(resp.guid, resp, False)
            return resp
        return orig(action, data, reasoning)
    env.step = step
    return env
