"""
Check that FindGoal layouts are reproducible: 30 resets with the same seed must give the
same walls and goal.

Run:  python tests/test_layout.py
"""
import gymnasium as gym
import stable_marl.envs
from stable_marl.core.constants import Type
env = gym.make('MultiGrid-FindGoal-15x15-v0', agents=2, render_mode='rgb_array',
               num_obstacles=6, width=15, height=15)

def wall_set(env):
    u = env.unwrapped
    return {(x, y) for x in range(u.grid.width) for y in range(u.grid.height)
            if (o := u.grid.get(x, y)) is not None and o.type == Type.wall}

layouts = []
for _ in range(30):
    env.reset(seed=0)
    layouts.append((wall_set(env), tuple(env.unwrapped.goal_pos)))

ref = layouts[0][0]
for k, (w, g) in enumerate(layouts):
    diff = ref ^ w   # symmetric difference
    if diff:
        print(f"reset {k}: goal={g}, {len(diff)} wall cells differ: {sorted(diff)}")
identical = sum(1 for w, g in layouts if w == ref and g == layouts[0][1])
print("identical resets:", identical, "/ 30")
if identical != len(layouts):
    raise SystemExit("FAILED: resets with the same seed produced different layouts")