"""
VMAS scenarios (navigation, transport, balance, flocking, football, the MPE tasks, ...) as stable-marl
envs: vector observations, continuous or discrete actions, and the scenario's infos (reward terms).
Needs `pip install 'stable-marl[vmas]'`.

    python scripts/examples/vmas_scenarios.py

Ids: VMAS-<Scenario>-v0 (e.g. VMAS-Navigation-v0, VMAS-SimpleSpread-v0), with the scenario's arguments
(e.g. n_agents), `continuous_actions` and `max_steps` as arguments.
"""
import tempfile

import numpy as np

import stable_marl as sm

ids = sm.list_envs('vmas')
print(f'{len(ids)} VMAS envs, e.g.', ids[:4])

# 1. the env itself: observations, actions, infos and state keyed by agent id
env = sm.make('VMAS-Navigation-v0', n_agents=3, max_steps=50)
obs, infos = env.reset(seed=0)
print('agent 0 observes', obs[0]['vector'].shape, '| actions', env.action_space[0], '| infos', sorted(infos[0]))
print('state (position, velocity of every entity):', env.state().shape)
returns = np.zeros(3)
for t in range(50):
    obs, rewards, terminations, truncations, infos = env.step({a: env.action_space[a].sample() for a in range(3)})
    returns += [rewards[a] for a in range(3)]
    if all(terminations.values()) or all(truncations.values()):
        break
print(f'random actions for {t + 1} steps: returns {np.round(returns, 2).tolist()}')

# discrete actions (VMAS's 9 moves, 0: none)
env = sm.make('VMAS-Navigation-v0', n_agents=3, continuous_actions=False)
print('discrete actions:', env.action_space[0], '| no-op', env.noop_action)

# agents of different kinds: observations zero-padded, actions padded with dimensions bounded to 0
env = sm.make('VMAS-SimpleSpeakerListener-v0')
print(f'speaker / listener: observation sizes {env.obs_dims} (padded to {env.observation_space[0]["vector"].shape[0]}), '
      f'action sizes {env.action_dims} (padded to {env.action_space[0].shape[0]})')

# 2. in a World: a dataset (observations, actions, rewards, the scenario's infos, the state)
with tempfile.TemporaryDirectory() as tmp:
    world = sm.World('VMAS-Navigation-v0', num_envs=4, n_agents=3, max_steps=50)
    world.set_policy(sm.RandomPolicy(seed=0))
    world.collect(f'{tmp}/navigation.h5', episodes=8, seed=0, progress=False)
    ds = sm.HDF5Dataset(path=f'{tmp}/navigation.h5', num_steps=4)
    print('dataset columns:', ds.column_names)
    print('a clip:', {k: tuple(v.shape) for k, v in ds[0].items() if k in ('vector', 'action', 'reward', 'pos_rew', 'state')})

# 3. random policies on a few scenarios. `success_rate` counts the episodes the scenario ended before the step
#    limit: a success in some (navigation: every agent on its goal), a failure in others (balance: the package fell)
for env_id, kwargs in (('VMAS-Navigation-v0', {'n_agents': 2}), ('VMAS-Transport-v0', {'n_agents': 4}),
                       ('VMAS-Balance-v0', {'n_agents': 3}), ('VMAS-SimpleSpread-v0', {})):
    world = sm.World(env_id, num_envs=4, max_steps=100, **kwargs)
    world.set_policy(sm.RandomPolicy(seed=0))
    r = world.evaluate(episodes=8, seed=0)
    print(f"{env_id:22s} {world.num_agents} agents: mean return {r['mean_return']:8.2f}, ended by the scenario {r['success_rate']:5.1f}%, "
          f"mean length {r['mean_length']:.0f}")
