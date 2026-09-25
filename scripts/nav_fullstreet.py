"""Full-street run of the navigation controllers for the viewer's main robot.

Same street, start and goal as the Isaac M4 runs (camera path start -> end), crowd from
crowd_world_v2 (corrected radii), CrowdEnv without random start: the robot starts where
the phone started at t=0 and heads where it went. Controllers: the trained PPO policy
(nav_policy.zip) and the rule-based TTC controller, same crowd, same clock.
Robot at velocity-command level (what the controllers output; the G1 locomotion policy
turns that into steps in Isaac -- not simulated here).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from densewalk.crowd_env import CrowdEnv  # noqa: E402
from nav_compare_export import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--crowd", required=True)
ap.add_argument("--policy", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=45.0)
ap.add_argument("--reverse", action="store_true",
                help="start at the far end and walk back against the crowd flow (most people move +x)")
a = ap.parse_args()
from stable_baselines3 import PPO
model = PPO.load(a.policy, device="cpu")
res = {}
class ReverseEnv(CrowdEnv):
    """Same env, start and goal swapped: the robot meets the crowd head-on."""
    def reset(self, seed=None):
        super().reset(seed)
        self.pos, self.goal_xy = self.cam_path[-1].astype(float).copy(), self.cam_path[0].astype(float).copy()
        d = self.goal_xy - self.pos
        self.yaw = float(np.arctan2(d[1], d[0])); self.vel = np.zeros(2)
        self.prev_goal_dist = float(np.linalg.norm(d))
        return self._obs()


for key in ("learned", "rule"):
    Env = ReverseEnv if a.reverse else CrowdEnv
    env = Env(a.crowd, seconds=a.seconds, seed=0, randomize_start=False)
    log = run(env, 0, key, model)
    e = log["end"]
    res[key] = log | {"outcome": "reached" if e["reached"] else "collided" if e["collided"] else "left street" if e["out"] else "timeout"}
    print(key, res[key]["outcome"], "steps", len(log["steps"]), "min clearance", log["min_clear"], "goal dist", e["goal_dist"])
Path(a.out).write_text(json.dumps({"what": __doc__.strip().splitlines()[0], "dt": env.dt, **res}, separators=(",", ":")))
