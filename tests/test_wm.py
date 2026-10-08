"""
Tests of the decentralized LeWM world models (stable_marl.wm): training on collected data, and each
agent planning with its own model from its own view (`pixels`).

    python tests/test_wm.py
"""
import sys
import tempfile
import traceback

import numpy as np
import torch

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_marl.wm import DecentralizedWorldModel, train_world_model

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn

ENV = dict(agents=2, tile_size=8, size=7, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1)
TINY = dict(image_size=32, patch_size=8, embed_dim=32, depth=2, heads=2, dim_head=16, mlp_dim=64, projector_hidden=64,
            encoder_kwargs=dict(dim=32, depth=2, heads=2, mlp_dim=64), history_size=3, dropout=0.0)


def dataset(path):
    for policy, episodes, seed in ((GoToGoalPolicy(), 24, 0), (sm.RandomPolicy(seed=0), 8, 1000)):
        world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, max_episode_steps=20, goal_conditioned=True, **ENV)
        world.set_policy(policy)
        world.collect(path, episodes=episodes, seed=seed, progress=False)
    return sm.HDF5Dataset(path=path, num_steps=4, keys_to_load=['pixels', 'action', 'terminated'])


@test
def train_then_plan_per_agent():
    with tempfile.TemporaryDirectory() as tmp:
        ds = dataset(f'{tmp}/d.h5')
        torch.manual_seed(0)
        model = DecentralizedWorldModel.build_lewm(num_agents=2, num_actions=4, **TINY)
        history = train_world_model(model, ds, epochs=3, batch_size=32, lr=1e-3, sigreg_kwargs={'num_proj': 64},
                                    log=None)
        assert history['train_loss'][-1] < history['train_loss'][0]
        model.save(f'{tmp}/wm.pt')
        model = DecentralizedWorldModel.load(f'{tmp}/wm.pt')

    seen = []
    rollout = model.rollout
    def spy(info_dict, candidates):
        seen.append((tuple(info_dict['pixels'].shape), tuple(candidates.shape)))
        return rollout(info_dict, candidates)
    model.rollout = spy

    cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True))
    solver = CategoricalCEMSolver(cost, batch_size=2, num_samples=16, n_steps=2, topk=4, mode='per_agent')
    policy = sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=4, receding_horizon=2, history_len=3))
    assert not model.training                                   # planning in inference mode
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=8, goal_conditioned=True, **ENV)
    world.set_policy(policy)
    res = world.evaluate(episodes=2, seed=0)
    assert len(res['episode_lengths']) == 2 and seen
    frames = {s[0][2] for s in seen}
    assert seen[0][0][3:] == (2, 56, 56, 3) and max(frames) == 3      # each agent's view, up to 3 context frames
    assert all(s[1][2:] == (4, 2, 4) for s in seen)                    # (horizon, agents, one-hot actions)


@test
def position_as_extra_input():
    """Each agent's model also reads its own position (and the goal's): train, then plan with it."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset(f'{tmp}/d.h5')
        ds = sm.HDF5Dataset(path=f'{tmp}/d.h5', num_steps=4, keys_to_load=['pixels', 'position', 'action', 'terminated'])
        torch.manual_seed(0)
        model = DecentralizedWorldModel.build_lewm(num_agents=2, num_actions=4, extra_inputs={'position': 2}, **TINY)
        train_world_model(model, ds, epochs=1, batch_size=32, lr=1e-3, sigreg_kwargs={'num_proj': 64}, log=None)

    def policy(history_keys):
        cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True, step_reduction='min'))
        solver = CategoricalCEMSolver(cost, batch_size=2, num_samples=16, n_steps=2, topk=4, mode='per_agent')
        return sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=4, receding_horizon=2, history_len=3),
                                   history_keys=history_keys)

    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=8, goal_conditioned=True, **ENV)
    world.set_policy(policy(('pixels', 'position')))
    assert len(world.evaluate(episodes=2, seed=0)['episode_lengths']) == 2
    world.set_policy(policy(('pixels',)))        # position given for 1 frame, pixels for up to 3: a clear error
    try:
        world.evaluate(episodes=2, seed=0)
        raise AssertionError("missing position history must fail")
    except ValueError as e:
        assert 'history_keys' in str(e)


@test
def agents_plan_independently():
    """An agent's planned actions do not depend on what the other agent sees."""
    torch.manual_seed(0)
    model = DecentralizedWorldModel.build_lewm(num_agents=2, num_actions=4, **TINY).eval()
    cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True))
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=1, goal_conditioned=True, **ENV)
    world.reset(seed=3)
    plans = []
    for blank_agent_1 in (False, True):
        info = {k: torch.as_tensor(v) for k, v in world.infos.items() if isinstance(v, np.ndarray) and v.dtype != object}
        if blank_agent_1:
            info['pixels'] = info['pixels'].clone(); info['pixels'][:, :, 1] = 0
            info['goal'] = info['goal'].clone(); info['goal'][:, :, 1] = 0
        solver = CategoricalCEMSolver(cost, num_samples=32, n_steps=3, topk=8, mode='per_agent', seed=0)
        solver.configure(action_space=world.envs.single_action_space, n_envs=1,
                         config=sm.PlanConfig(horizon=4, receding_horizon=1))
        plans.append(solver.solve(info)['actions'])
    assert torch.equal(plans[0][..., 0, :], plans[1][..., 0, :])        # agent 0's plan is unchanged


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
    print(f'All {len(TESTS)} world model tests passed.' if not failed else f'{failed} / {len(TESTS)} world model tests failed.')
    sys.exit(1 if failed else 0)
