"""
Quickstart: run policies in a pool of multi-agent envs, evaluate them, and watch an episode.

    python scripts/examples/quickstart.py

`World` runs `num_envs` copies of a registered env (here FindGoal: every agent has to reach the
green goal) and keeps everything about the current step in one info dict, `world.infos`, with
shapes (num_envs, 1, ...) and an agent axis on per-agent keys, e.g. `pixels` (each agent's view)
is (num_envs, 1, num_agents, H, W, 3).
"""
from pathlib import Path

import stable_marl as sm
from stable_marl.data import ReplayBuffer
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.utils import record_video_from_dataset

VIDEO_DIR = Path(__file__).parent / 'videos' / 'quickstart'
ENV = dict(agents=2, size=9, num_obstacles=2, n_clutter=0, tile_size=16)

print('registered envs:', len(sm.list_envs()), 'e.g.', sm.list_envs()[:3])

world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, max_episode_steps=50, goal_conditioned=True, **ENV)
world.reset(seed=0)
print({k: v.shape for k, v in world.infos.items() if hasattr(v, 'shape') and k in ('pixels', 'goal', 'position', 'state')})

# evaluate a random policy and the expert (a shortest-path policy for navigation envs)
for name, policy in (('random', sm.RandomPolicy(seed=0)), ('expert', GoToGoalPolicy())):
    world.set_policy(policy)
    results = world.evaluate(episodes=20, seed=0)
    print(f"{name:7s} success {results['success_rate']:5.1f}%  mean length {results['mean_length']:5.1f}")

# watch the expert: collect two episodes in memory, then save each agent's view side by side
world.set_policy(GoToGoalPolicy())
buffer = ReplayBuffer(max_steps=1_000)
world.collect(writer=buffer, episodes=2, seed=0, progress=False)
files = record_video_from_dataset(VIDEO_DIR, buffer, [0, 1], fps=4)
print('videos:', [str(f) for f in files])
