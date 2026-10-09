"""
The multi-robot warehouse (RWARE): robots fetch requested shelves, deliver them to the goal cells and
bring them back; each delivery is rewarded. Each robot observes the square around it as RWARE's
vector (`vector`) and as an RGB image (`pixels`).

    python scripts/examples/rware_warehouse.py

Ids: RWARE-Tiny-v0, RWARE-Small-v0, RWARE-Medium-v0, RWARE-Large-v0 (RWARE's warehouse sizes), with
the number of robots, the difficulty (requested shelves per robot), the sensor range, the step limit,
the reward type and the observations as arguments.
"""
import tempfile
from pathlib import Path

import numpy as np

import stable_marl as sm

VIDEO_DIR = Path(__file__).parent / 'videos' / 'rware'

print('RWARE envs:', sm.list_envs('rware'))

# 1. the env itself: per-robot observations keyed by robot id, and a global state
env = sm.make('RWARE-Tiny-v0', agents=2, sensor_range=1)
obs, infos = env.reset(seed=0)
print(f'warehouse {env.height} x {env.width}, actions {env.action_space[0]} (noop, forward, left, right, load/unload)')
print('robot 0 observes:', {k: getattr(v, 'shape', v) for k, v in obs[0].items()})
print('global state:', env.state().shape, '(shelf, requested, robot, heading, carrying, goal, corridor)')

# random robots for one episode: deliveries are rare (the reward is sparse)
deliveries, done = 0.0, False
while not done:
    obs, rewards, terminations, truncations, infos = env.step({a: env.action_space[a].sample() for a in range(2)})
    deliveries += sum(rewards.values())
    done = all(truncations.values())
print(f'random robots, one episode of {env.warehouse.max_steps} steps: {deliveries:.0f} deliveries')

# 2. in a World: a dataset with both observations, plus a render of the whole warehouse (`render`)
with tempfile.TemporaryDirectory() as tmp:
    world = sm.World('RWARE-Tiny-v0', num_envs=4, agents=2, max_steps=100, add_pixels=True)
    world.set_policy(sm.RandomPolicy(seed=0))
    world.collect(f'{tmp}/rware.h5', episodes=8, seed=0, progress=False)
    ds = sm.HDF5Dataset(path=f'{tmp}/rware.h5', num_steps=4)
    print(f'dataset: {len(ds.lengths)} episodes;', {k: tuple(v.shape) for k, v in ds[0].items()
                                                    if k in ('vector', 'pixels', 'render', 'action', 'reward')})

# 3. vector observations only (no images: faster), a bigger warehouse with more robots
world = sm.World('RWARE-Small-v0', num_envs=4, agents=4, difficulty='easy', observations=['vector'], max_steps=200)
world.set_policy(sm.RandomPolicy(seed=0))
results = world.evaluate(episodes=8, seed=0)
print(f"RWARE-Small, 4 random robots: {results['mean_return']:.2f} deliveries per episode;",
      'observation keys:', sorted(k for k in world.infos if k in ('vector', 'pixels', 'direction', 'position', 'carrying')))

# 4. a video of the whole warehouse per episode
world = sm.World('RWARE-Tiny-v0', num_envs=1, agents=2, max_steps=60, add_pixels=True, tile_size=16)
world.set_policy(sm.RandomPolicy(seed=0))
world.evaluate(episodes=1, seed=0, video=VIDEO_DIR)
print('video:', sorted(str(p) for p in VIDEO_DIR.glob('*.mp4')))
