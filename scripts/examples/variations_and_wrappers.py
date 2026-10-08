"""
Factors of variation and visual perturbations: change colors, layouts and the agents' views, to
test how robust a model or policy is.

    python scripts/examples/variations_and_wrappers.py

Variations (as in stable-worldmodel) are reset options: `variation` lists factors to sample at each
reset (or 'all'), `variation_values` sets them. Their values are added to the infos (and to collected
datasets) as `variation.<factor>`. Visual wrappers transform each agent's view.
"""
from functools import partial
from pathlib import Path

import numpy as np

import stable_marl as sm
from stable_marl.data import ReplayBuffer
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.plot import save_panel_videos
from stable_marl.wrappers import GrayscaleWrapper, NoiseWrapper, OcclusionWrapper, linear

VIDEO_DIR = Path(__file__).parent / 'videos' / 'variations'
ENV = dict(agents=2, size=9, num_obstacles=3, n_clutter=0, tile_size=16)

env = sm.make('MultiGrid-FindGoal-15x15-v0', **ENV)
print('factors:', env.variation_space.names())

world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=40, **ENV)
world.reset(seed=0, options={'variation': ['wall.color', 'agent.color']})          # sampled at each reset
print('sampled wall colors:', world.infos['variation.wall.color'][:, 0].tolist())
world.reset(seed=0, options={'variation_values': {'obstacles.number': 0, 'goal.min_spawn_distance': 6}})
print('set values:', world.infos['variation.obstacles.number'][:, 0].tolist())

# how much does the expert care? (it plans on the true state, so it should not)
world.set_policy(GoToGoalPolicy())
print('expert, all factors varied:', world.evaluate(episodes=10, seed=0, options={'variation': ['all']})['success_rate'], '%')

# perturbed views: noise growing over the episode, occlusions, grayscale (after MegaWrapper: on the infos)
views = {}
for name, wrapper in (('clean', None), ('noise', partial(NoiseWrapper, std=linear(0, 60, 20), seed=0)),
                      ('occlusion', partial(OcclusionWrapper, num_patches=2, size=(0.2, 0.35), seed=0)),
                      ('grayscale', GrayscaleWrapper)):
    w = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=1, max_episode_steps=20, **ENV,
                 extra_wrappers=[wrapper] if wrapper else None)
    w.set_policy(GoToGoalPolicy())
    buffer = ReplayBuffer(max_steps=100)
    w.collect(writer=buffer, episodes=1, seed=3, progress=False)
    views[name] = [np.stack(next(buffer.episodes())['pixels'])]          # one env: (T, agents, H, W, 3)
save_panel_videos(VIDEO_DIR, views, fps=4)
print('panel video:', VIDEO_DIR / 'env_0.mp4')
