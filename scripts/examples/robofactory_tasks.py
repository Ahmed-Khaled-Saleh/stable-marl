"""
RoboFactory: 2 to 4 Panda arms solving manipulation tasks together (ManiSkill 3). Each arm observes its own
camera (`pixels`) and its joint positions and velocities (`vector`); the reward is 1 when the task is done.

    python scripts/examples/robofactory_tasks.py

Needs ManiSkill (`pip install 'stable-marl[robofactory]'`); the 3D assets (40 MB) are downloaded on first
use. The last part, RoboFactory's motion-planning expert, needs numpy<2 (mplib 0.1.1) and is skipped otherwise.

Ids: RoboFactory-<task>-v0 for the tasks LiftBarrier, PassShoe, PlaceFood, TwoRobotsStackCube (2 arms),
CameraAlignment, ThreeRobotsStackCube (3), TakePhoto, LongPipelineDelivery (4), with the scene, observations,
camera size and step limit as arguments.
"""
import tempfile
from pathlib import Path

import numpy as np

import stable_marl as sm
from stable_marl.envs.robofactory import TASKS, RoboFactoryExpert

VIDEO_DIR = Path(__file__).parent / 'videos' / 'robofactory'

print('RoboFactory envs:', sm.list_envs('robofactory'))
print('arms per task:', {task: arms for task, (_, arms) in TASKS.items()})

# 1. the env itself: per-arm observations keyed by arm id, the simulation state, a global camera
env = sm.make('RoboFactory-LiftBarrier-v0', camera_size=(64, 64))
obs, infos = env.reset(seed=0)
print(f'{env.task}: {env.num_agents} arms, step limit {env.max_steps}, action of arm 0 {env.action_space[0].shape} '
      '(pd_joint_pos: 7 joint targets + gripper)')
print('arm 0 observes:', {k: v.shape for k, v in obs[0].items()}, '| infos:', infos[0])
print('global state:', env.state().shape, '(poses and velocities of the objects and arms) | render:', env.render().shape)

# random arms: the joint targets jump around; flung high enough, the barrier counts as lifted
for t in range(1, 101):
    obs, rewards, terminations, truncations, infos = env.step({a: env.action_space[a].sample() for a in range(2)})
    if terminations[0]:
        break
print(f'random arms: {"task done" if terminations[0] else "not done"} after {t} steps (reward {rewards[0]:.0f})')
env.close()

# 2. in a World: a dataset with both observations and the global camera (`render`), then a dataset-driven
#    evaluation starting from recorded steps (the env restores the recorded simulation state)
with tempfile.TemporaryDirectory() as tmp:
    world = sm.World('RoboFactory-PassShoe-v0', num_envs=2, max_episode_steps=20, camera_size=(64, 64),
                     render_size=(96, 128), add_pixels=True)
    world.set_policy(sm.RandomPolicy(seed=0))
    world.collect(f'{tmp}/passshoe.h5', episodes=2, seed=0, progress=False)
    ds = sm.HDF5Dataset(path=f'{tmp}/passshoe.h5', num_steps=4)
    print(f'dataset: {len(ds.lengths)} episodes;', {k: tuple(v.shape) for k, v in ds[0].items()
                                                    if k in ('vector', 'pixels', 'render', 'action', 'reward', 'state')})
    res = world.evaluate(dataset=ds, episodes_idx=[0, 1], start_steps=[5, 10], goal_offset=5, eval_budget=10)
    print('dataset-driven evaluation (random policy):', res['success_rate'], '% success,',
          'goal images in the infos:', world.infos['goal'].shape)

# 3. vector observations only (no cameras rendered: faster), four arms
world = sm.World('RoboFactory-TakePhoto-v0', num_envs=2, observations=['vector'], max_episode_steps=50)
world.set_policy(sm.RandomPolicy(seed=0))
results = world.evaluate(episodes=2, seed=0)
print(f"TakePhoto, 4 random arms: {results['success_rate']:.0f}% success; vector {world.infos['vector'].shape}")

# 4. RoboFactory's motion-planning expert (numpy<2 only), with a video of the global camera per episode
if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
    print('numpy', np.__version__, '>= 2: skipping the expert (mplib 0.1.1 needs numpy<2)')
else:
    world = sm.World('RoboFactory-LiftBarrier-v0', num_envs=1, camera_size=(64, 64), render_size=(240, 320), add_pixels=True)
    expert = RoboFactoryExpert()
    world.set_policy(expert)
    results = world.evaluate(episodes=2, seed=0, video=VIDEO_DIR)
    print(f"LiftBarrier, motion-planning expert: {results['success_rate']:.0f}% success, "
          f"episode lengths {results['episode_lengths'].tolist()}")
    print('videos:', sorted(str(p) for p in VIDEO_DIR.glob('*.mp4')))
