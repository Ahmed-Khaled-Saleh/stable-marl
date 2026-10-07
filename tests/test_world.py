"""
Tests of the World / EnvPool / policy / dataset pipeline (stable_marl.world, .policy, .data).

    python tests/test_world.py
"""
import sys
import tempfile
import traceback

import numpy as np

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.world import EnvPool
from stable_marl.wrappers.default import MegaWrapper

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


def collect(path, env_name='MultiGrid-Empty-6x6-v0', num_envs=3, episodes=6, seed=0, policy=None, **kwargs):
    world = sm.World(env_name, num_envs=num_envs, **{'agents': 2, **kwargs})
    world.set_policy(policy or sm.RandomPolicy(seed=0))
    world.collect(path, episodes=episodes, seed=seed, progress=False)
    return sm.HDF5Dataset(path=path)


def episode_columns(ds, e):
    "The raw (numpy) columns of episode `e`."
    start, length = int(ds.offsets[e]), int(ds.lengths[e])
    return {col: ds.get_col_data(col)[start:start + length] for col in ds.column_names}


@test
def collected_episodes_replay_exactly():
    """Re-running the stored actions from the stored seed reproduces every stored row."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = collect(f'{tmp}/d.h5', 'MultiGrid-FindGoal-15x15-v0', episodes=4, seed=3, num_obstacles=6)
        meta = ds.metadata
        for e in range(len(ds.lengths)):
            ep = episode_columns(ds, e)
            T = len(ep['step_idx']) - 1
            env = sm.make(meta['env_name'], **meta['env_kwargs'])
            obs, _ = env.reset(seed=int(ep['seed'][0]))
            assert np.isnan(ep['reward'][0]).all() and np.isnan(ep['action'][T]).all()
            assert list(ep['step_idx']) == list(range(T + 1)) and len(set(ep['id'])) == 1
            for t in range(T + 1):
                for a in range(2):
                    assert np.array_equal(ep['pov'][t, a], obs[a]['pov']), (e, t, a)
                    assert np.array_equal(ep['image'][t, a], obs[a]['image']), (e, t, a)
                assert np.array_equal(ep['state'][t], env.state())
                if t == T:
                    break
                obs, rew, term, trunc, _ = env.step({a: int(ep['action'][t, a]) for a in range(2)})
                assert all(ep['reward'][t + 1, a] == np.float32(rew[a]) and ep['terminated'][t + 1, a] == term[a]
                           for a in range(2))
            assert (term[0] and term[1]) or trunc[0], "a stored episode ends with the env's episode"


@test
def collect_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = collect(f'{tmp}/a.h5', seed=7), collect(f'{tmp}/b.h5', seed=7)
        assert np.array_equal(a.lengths, b.lengths) and a.column_names == b.column_names
        for col in a.column_names:
            assert np.array_equal(a.get_col_data(col), b.get_col_data(col), equal_nan=col in ('action', 'reward')), col


@test
def episodes_are_seeded_by_index():
    with tempfile.TemporaryDirectory() as tmp:
        ds = collect(f'{tmp}/d.h5', episodes=7, seed=100)
        seeds = sorted(int(episode_columns(ds, e)['seed'][0]) for e in range(7))
        assert seeds == list(range(100, 107))


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
    for env_name, kwargs in [('MultiGrid-Empty-8x8-v0', {}), ('MultiGrid-FindGoal-15x15-v0', {'num_obstacles': 6})]:
        world = sm.World(env_name, num_envs=2, agents=2, **kwargs)
        world.set_policy(GoToGoalPolicy())
        res = world.evaluate(episodes=6, seed=0)
        assert res['success_rate'] == 100.0, (env_name, res['episode_successes'])


@test
def random_policy_is_reproducible():
    runs = []
    for _ in range(2):
        world = sm.World('MultiGrid-Empty-6x6-v0', num_envs=3, agents=2)
        world.set_policy(sm.RandomPolicy(seed=42))
        runs.append(world.evaluate(episodes=5, seed=0)['episode_lengths'])
    assert np.array_equal(*runs)


@test
def world_options():
    world = sm.World('MultiGrid-Empty-6x6-v0', num_envs=2, agents=2, max_episode_steps=5, add_pixels=True)
    world.set_policy(sm.RandomPolicy(seed=0))
    res = world.evaluate(episodes=3, seed=0)
    assert (res['episode_lengths'] <= 5).all() and world.infos['pixels'].shape == (2, 1, 192, 192, 3)
    assert len(world.evaluate(episodes=10, seed=0, reset_mode='wait')['seeds']) == 2


@test
def pool_mask_and_reset():
    pool = EnvPool([lambda: MegaWrapper(sm.make('MultiGrid-Empty-6x6-v0', agents=2))] * 3)
    try:
        pool.reset(seed=0, mask=np.array([True, False, False]))
        raise AssertionError("a partial first reset must fail")
    except RuntimeError:
        pass
    _, infos = pool.reset(seed=[5, 6, 7])
    before = infos['image'].copy()
    _, infos = pool.reset(seed=[0, 0, 9], mask=np.array([False, False, True]))
    assert list(pool.seeds) == [5, 6, 9] and np.array_equal(infos['image'][:2], before[:2])
    assert list(infos['seed'][:, 0]) == [5, 6, 9]


@test
def single_env_info_matches_world():
    """A single env wrapped with MegaWrapper gives the info rows the World records."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = collect(f'{tmp}/d.h5', num_envs=1, episodes=1, seed=4)
        ep = episode_columns(ds, 0)
        env = MegaWrapper(sm.make('MultiGrid-Empty-6x6-v0', agents=2))
        _, info = env.reset(seed=4)
        for t in range(len(ep['step_idx'])):
            for col in ds.column_names:
                assert np.array_equal(ep[col][t], info[col], equal_nan=True) or col == 'action', (t, col)
            if t + 1 < len(ep['step_idx']):
                _, _, _, _, info = env.step({a: int(ep['action'][t, a]) for a in range(2)})


@test
def dataset_clips():
    with tempfile.TemporaryDirectory() as tmp:
        collect(f'{tmp}/d.h5', episodes=3, seed=1)
        ds = sm.HDF5Dataset(path=f'{tmp}/d.h5', num_steps=4, keys_to_load=['pov', 'state', 'action', 'step_idx'])
        assert len(ds) == sum(max(int(n) - 4 + 1, 0) for n in ds.lengths)
        item = ds[len(ds) - 1]
        assert item['pov'].shape == (4, 2, 224, 224, 3) and item['action'].shape == (4, 2)
        assert item['state'].shape == (4, 3, 6, 6) and set(item) == {'pov', 'state', 'action', 'step_idx'}
        assert list(item['step_idx'].numpy()) == list(range(int(ds.lengths[-1]) - 4, int(ds.lengths[-1])))


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
