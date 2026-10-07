"""
Tests of the World / EnvPool / data pipeline (stable_marl.world, stable_marl.policy, stable_marl.data).

    python tests/test_world.py
"""
import sys
import tempfile
import traceback
from functools import partial

import numpy as np

import stable_marl as sm
from stable_marl.data import NpzEpisodes
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.world import EnvPool

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


def collect(path, env_id='MultiGrid-Empty-6x6-v0', num_envs=3, episodes=6, seed=0, policy=None, **env_kwargs):
    world = sm.World(env_id, num_envs=num_envs, **{'agents': 2, **env_kwargs})
    world.set_policy(policy or sm.RandomPolicy(seed=0))
    world.collect(path, episodes=episodes, seed=seed, progress=False)
    return sm.load_episodes(path)


@test
def collected_episodes_replay_exactly():
    """Re-running the stored actions from the stored seed reproduces every stored step."""
    with tempfile.TemporaryDirectory() as tmp:
        data = collect(tmp, 'MultiGrid-FindGoal-15x15-v0', episodes=4, seed=3, num_obstacles=6)
        meta = data.metadata
        for e in range(data.num_episodes):
            ep = data.episode(e)
            env = sm.make(meta['env_id'], **meta['env_kwargs'])
            obs, _ = env.reset(seed=int(ep['seed']))
            assert np.array_equal(ep['obs.image'][0], np.stack([obs[a]['image'] for a in range(2)]))
            for t in range(int(ep['length'])):
                obs, rew, term, trunc, _ = env.step({a: int(ep['action'][t, a]) for a in range(2)})
                for a in range(2):
                    assert np.array_equal(ep['obs.image'][t + 1, a], obs[a]['image']), (e, t, a)
                    assert np.array_equal(ep['obs.pov'][t + 1, a], obs[a]['pov']), (e, t, a)
                    assert ep['reward'][t, a] == np.float32(rew[a]) and ep['terminated'][t, a] == term[a]
                assert np.array_equal(ep['state'][t + 1], env.state())
            assert (term[0] and term[1]) or trunc[0], "stored episode must end with the env's episode"


@test
def collect_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        da, db = collect(a, seed=7), collect(b, seed=7)
        for e in range(da.num_episodes):
            ea, eb = da.episode(e), db.episode(e)
            assert ea.keys() == eb.keys() and all(np.array_equal(ea[k], eb[k]) for k in ea)


@test
def episodes_are_indexed_and_seeded():
    with tempfile.TemporaryDirectory() as tmp:
        data = collect(tmp, episodes=7, seed=100)
        pairs = sorted((int(data.episode(e)['episode_index']), int(data.episode(e)['seed']))
                       for e in range(data.num_episodes))
        assert pairs == [(k, 100 + k) for k in range(7)]


@test
def evaluate_does_not_depend_on_num_envs():
    results = []
    for num_envs in (1, 2, 5):
        world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=num_envs, agents=2, num_obstacles=6)
        world.set_policy(GoToGoalPolicy())
        results.append(world.evaluate(episodes=6, seed=0))
    for r in results[1:]:
        for key in ('episode_returns', 'episode_lengths', 'episode_successes', 'seeds'):
            assert np.array_equal(results[0][key], r[key]), key


@test
def expert_solves_navigation_envs():
    for env_id, kwargs in [('MultiGrid-Empty-8x8-v0', {}), ('MultiGrid-FindGoal-15x15-v0', {'num_obstacles': 6})]:
        world = sm.World(env_id, num_envs=2, agents=2, **kwargs)
        world.set_policy(GoToGoalPolicy())
        res = world.evaluate(episodes=6, seed=0)
        assert res['success_rate'] == 1.0, (env_id, res['episode_successes'])


@test
def pool_mask_and_reset():
    pool = EnvPool([partial(sm.make, 'MultiGrid-Empty-6x6-v0', agents=2)] * 3)
    try:
        pool.reset(seed=0, mask=np.array([True, False, False]))
        raise AssertionError("a partial first reset must fail")
    except RuntimeError:
        pass
    obs, _ = pool.reset(seed=[5, 6, 7])
    before = obs['image'].copy()
    obs, _ = pool.reset(seed=[0, 0, 9], mask=np.array([False, False, True]))
    assert list(pool.seeds) == [5, 6, 9] and np.array_equal(obs['image'][:2], before[:2])
    assert pool.state(np.array([True, False, True])).shape == (2, 6, 6, 3)


@test
def trajectory_dataset_windows():
    with tempfile.TemporaryDirectory() as tmp:
        data = collect(tmp, episodes=3, seed=1)
        ds = sm.TrajectoryDataset(tmp, num_steps=4, keys=['obs.image', 'state', 'action', 'reward'])
        lengths = [int(data.episode(e)['length']) for e in range(3)]
        assert len(ds) == sum(max(T + 2 - 4, 0) for T in lengths)
        item = ds[len(ds) - 1]
        assert item['obs.image'].shape == (4, 2, 7, 7, 3) and item['action'].shape == (3, 2)
        e, t = int(item['episode']), int(item['start'])
        assert np.array_equal(item['state'], data.episode(e)['state'][t:t + 4])
        assert isinstance(ds.episodes, NpzEpisodes) and 'obs.pov' not in item


if __name__ == '__main__':
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f'  OK  {fn.__name__}')
        except Exception:
            failed += 1
            print(f'FAIL  {fn.__name__}')
            traceback.print_exc()
    print(f'All {len(TESTS)} world tests passed.' if not failed else f'{failed} / {len(TESTS)} world tests failed.')
    sys.exit(1 if failed else 0)
