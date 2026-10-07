"""
End-to-end tests of model-based planning (stable_marl.planning, WorldModelPolicy), joint and per agent.

    python tests/test_planning.py
"""
import sys
import traceback

import numpy as np
import torch

import stable_marl as sm
from stable_marl.envs.multigrid.dynamics import NavigationDynamics
from stable_marl.planning import (CategoricalCEMSolver, CategoricalMPPISolver, GoalMSE, ShootingCostEvaluator,
                                  WeightedSum, ControlPenalty)

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn

# FindGoal ends when *every* agent reached the goal ('all'), so both agents must be planned for
OPEN_FINDGOAL = dict(agents=2, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1)


def planner(mode, solver_cls=CategoricalCEMSolver, num_envs=2, **solver_kwargs):
    cost = ShootingCostEvaluator(NavigationDynamics(), GoalMSE(per_agent=mode == 'per_agent'))
    kwargs = dict(batch_size=num_envs, num_samples=128, n_steps=6, mode=mode, seed=0)
    kwargs.update(dict(topk=16) if solver_cls is CategoricalCEMSolver else dict(temperature=1.0))
    solver = solver_cls(cost, **{**kwargs, **solver_kwargs})
    return sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2))


def evaluate(policy, env='MultiGrid-FindGoal-15x15-v0', num_envs=2, episodes=4, **env_kwargs):
    world = sm.World(env, num_envs=num_envs, goal_conditioned=True, **(env_kwargs or OPEN_FINDGOAL))
    world.set_policy(policy)
    return world.evaluate(episodes=episodes, seed=0)


@test
def joint_planning_reaches_the_goal():
    res = evaluate(planner('joint'))
    assert res['success_rate'] == 100.0, res['episode_successes']


@test
def per_agent_planning_reaches_the_goal():
    res = evaluate(planner('per_agent'))
    assert res['success_rate'] == 100.0, res['episode_successes']


@test
def mppi_in_both_modes():
    for mode in ('joint', 'per_agent'):
        res = evaluate(planner(mode, CategoricalMPPISolver))
        assert res['success_rate'] == 100.0, (mode, res['episode_successes'])


@test
def planning_is_reproducible():
    a, b = evaluate(planner('per_agent')), evaluate(planner('per_agent'))
    assert np.array_equal(a['episode_lengths'], b['episode_lengths'])


@test
def plans_beat_random_actions():
    planned = evaluate(planner('joint'), 'MultiGrid-Empty-8x8-v0', agents=2)
    world = sm.World('MultiGrid-Empty-8x8-v0', num_envs=2, agents=2)
    world.set_policy(sm.RandomPolicy(seed=0))
    random = world.evaluate(episodes=4, seed=0)
    assert planned['success_rate'] == 100.0 and planned['mean_length'] < random['mean_length']


@test
def joint_and_per_agent_costs_compose():
    cost = WeightedSum([(1.0, GoalMSE(per_agent=True)), (0.01, ControlPenalty(per_agent=True))])
    solver = CategoricalCEMSolver(ShootingCostEvaluator(NavigationDynamics(), cost), batch_size=2,
                                  num_samples=128, n_steps=6, topk=16, mode='per_agent')
    res = evaluate(sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2)))
    assert res['success_rate'] == 100.0


@test
def mode_and_cost_must_agree():
    solver = CategoricalCEMSolver(ShootingCostEvaluator(NavigationDynamics(), GoalMSE()), num_samples=16,
                                  n_steps=1, topk=4, mode='per_agent')
    try:
        evaluate(sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=4, receding_horizon=1)), episodes=1)
        raise AssertionError("a joint cost must be rejected in per_agent mode")
    except ValueError as e:
        assert 'per-agent cost' in str(e)


@test
def unsupported_configs_fail_clearly():
    for cfg in (dict(history_len=2), dict(action_block=2)):
        policy = sm.WorldModelPolicy(planner('joint').solver, sm.PlanConfig(horizon=4, receding_horizon=1, **cfg))
        try:
            sm.World('MultiGrid-Empty-6x6-v0', agents=2).set_policy(policy)
            raise AssertionError(cfg)
        except NotImplementedError:
            pass


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
    print(f'All {len(TESTS)} planning tests passed.' if not failed else f'{failed} / {len(TESTS)} planning tests failed.')
    sys.exit(1 if failed else 0)
