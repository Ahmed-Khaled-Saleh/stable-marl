"""
Tests of the factors of variation (stable_marl.spaces, MultiGridEnv.variation_space) and goal infos.

    python tests/test_variations.py
"""
import sys
import tempfile
import traceback

import numpy as np

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.envs.multigrid.core.constants import Color, Type
from stable_marl.envs.multigrid.core.world_object import WorldObj
from stable_marl.wrappers.default import MegaWrapper

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn

FINDGOAL = dict(agents=2, num_obstacles=6)


def cells(env, obj_type):
    "Colors of the cells holding objects of `obj_type`."
    state = env.unwrapped.grid.state
    return state[state[..., WorldObj.TYPE] == obj_type.to_index()][:, WorldObj.COLOR]


@test
def variation_values_are_applied_then_restored():
    env = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL)
    red, grey, green = Color.red.to_index(), Color.grey.to_index(), Color.green.to_index()
    native_agents = [Color(c).to_index() for c in env.agent_states.color]
    env.reset(seed=0, options={'variation_values': {'wall.color': red, 'goal.color': Color.blue.to_index(),
                                                    'agent.color': [3, 4]}})
    assert (cells(env, Type.wall) == red).all() and (cells(env, Type.goal) == Color.blue.to_index()).all()
    assert [Color(c).to_index() for c in env.agent_states.color] == [3, 4]
    obs, _ = env.reset(seed=0)   # no options: back to the native env
    assert (cells(env, Type.wall) == grey).all() and (cells(env, Type.goal) == green).all()
    assert [Color(c).to_index() for c in env.agent_states.color] == native_agents
    ref, _ = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL).reset(seed=0)
    assert all(np.array_equal(obs[a]['pixels'], ref[a]['pixels']) for a in range(2))


@test
def recoloring_never_changes_shared_objects():
    """`Wall()` is a cached shared instance: recoloring a grid must not change it."""
    from stable_marl.envs.multigrid.core.world_object import Wall
    env = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL)
    for _ in range(3):
        env.reset(seed=0, options={'variation_values': {'wall.color': Color.blue.to_index()}})
        env.grid.get(0, 0)
    assert Wall().color == Color.grey and (Wall().encode() == Wall(Color.grey).encode())
    fresh = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL)
    fresh.reset(seed=0)
    assert (cells(fresh, Type.wall) == Color.grey.to_index()).all()


@test
def init_value_changes_every_episode():
    env = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL, init_value={'wall.color': Color.purple.to_index()})
    for seed in (0, 1):
        env.reset(seed=seed)
        assert (cells(env, Type.wall) == Color.purple.to_index()).all()


@test
def structural_variations_change_the_layout():
    def walls(n):
        env = sm.make('MultiGrid-FindGoal-15x15-v0', agents=2, num_obstacles=6, n_clutter=0)
        env.reset(seed=0, options={'variation_values': {'obstacles.number': n}})
        return len(cells(env, Type.wall))
    border = 2 * 15 + 2 * 13
    assert walls(0) == border and walls(12) > walls(6) > border


@test
def sampling_is_reproducible_and_seeded():
    def sampled(seed):
        env = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL)
        env.reset(seed=seed, options={'variation': ['all']})
        return {n: np.asarray(env._variation(n)).tolist() for n in env.variation_space.names()}
    assert sampled(3) == sampled(3)
    assert len({str(sampled(s)) for s in range(6)}) > 1
    for values in (sampled(s) for s in range(6)):
        assert len(set(values['agent.color'])) == 2   # agents keep distinct colors


@test
def goal_infos():
    env = MegaWrapper(sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL))
    _, info = env.reset(seed=2)
    raw = env.unwrapped
    for a, agent in enumerate(raw.agents):
        assert np.array_equal(info['goal'][a], raw.get_goal_state(agent, agent.view_size))
        assert np.array_equal(info['goal_position'][a], raw.goal_pos)
    _, _, _, _, step_info = env.step({0: 2, 1: 2})
    assert np.array_equal(step_info['goal'], info['goal'])   # constant within the episode
    _, info = MegaWrapper(sm.make('MultiGrid-RedBlueDoors-6x6-v0', agents=2)).reset(seed=0)
    assert 'goal' not in info                                 # envs without goals


@test
def varied_episodes_replay_from_the_dataset():
    """An episode collected with every factor varied replays from its seed and recorded factor values."""
    with tempfile.TemporaryDirectory() as tmp:
        world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, **FINDGOAL)
        world.set_policy(GoToGoalPolicy())
        world.collect(f'{tmp}/d.h5', episodes=4, seed=0, options={'variation': ['all']}, progress=False)
        ds = sm.HDF5Dataset(path=f'{tmp}/d.h5')
        names = [c[len('variation.'):] for c in ds.column_names if c.startswith('variation.')]
        assert sorted(names) == sorted(sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL).variation_space.names())
        for e in range(len(ds.lengths)):
            start, T = int(ds.offsets[e]), int(ds.lengths[e])
            col = lambda c: ds.get_col_data(c)[start:start + T]
            values = {n: col(f'variation.{n}')[0].tolist() for n in names}
            env = sm.make('MultiGrid-FindGoal-15x15-v0', **FINDGOAL)
            obs, _ = env.reset(seed=int(col('seed')[0]), options={'variation_values': values})
            for t in range(T):
                assert all(np.array_equal(col('pixels')[t, a], obs[a]['pixels']) for a in range(2)), (e, t)
                if t + 1 < T:
                    obs, *_ = env.step({a: int(col('action')[t, a]) for a in range(2)})


@test
def unreachable_spawn_distance_does_not_hang():
    """A goal spawn distance no free cell has (small grid, or sampled with variation=['all']) is lowered, with a warning."""
    import warnings
    env = sm.make('MultiGrid-FindGoal-15x15-v0', agents=2, size=5, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        env.reset(seed=0)
    assert any('no free cell is 10 steps' in str(w.message) for w in caught)
    goal = np.asarray(env.goal_pos)
    farthest = max(abs(x - goal[0]) + abs(y - goal[1]) for x in range(1, 4) for y in range(1, 4))
    assert all(np.abs(np.asarray(a.state.pos) - goal).sum() >= farthest - 1 for a in env.agents)   # as far as possible
    small = sm.make('MultiGrid-FindGoal-15x15-v0', agents=2, size=7, num_obstacles=2, n_clutter=0)
    for seed in range(20):                                   # sampled distances up to 10 on a 5 x 5 interior
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            small.reset(seed=seed, options={'variation': ['all']})


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
    print(f'All {len(TESTS)} variation tests passed.' if not failed else f'{failed} / {len(TESTS)} variation tests failed.')
    sys.exit(1 if failed else 0)
