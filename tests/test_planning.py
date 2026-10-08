"""
End-to-end tests of model-based planning (stable_marl.planning, WorldModelPolicy): discrete and
continuous actions, joint and per-agent planning, history and action blocks.

    python tests/test_planning.py
"""
import sys
import traceback
from functools import partial

import numpy as np
import torch
from gymnasium import spaces

import stable_marl as sm
from stable_marl.envs.base import MultiAgentEnv
from stable_marl.envs.multigrid.dynamics import NavigationDynamics
from stable_marl.planning import (PLANNING_MODES, CategoricalCEMSolver, CategoricalMPPISolver, CEMSolver, ControlPenalty,
                                  GoalMSE, GradientSolver, ICEMSolver, LagrangianSolver, MPPISolver,
                                  PredictiveSamplingSolver, ShootingCostEvaluator, WeightedSum, flat_goal_encode)

TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


# ------------------------------------------------------------------------------------------- helpers
class Recording(torch.nn.Module):
    "Wraps a Dynamics model and records the shapes of the context it is given."
    def __init__(self, model, keys):
        super().__init__()
        self.model, self.keys, self.seen = model, keys, []

    def encode(self, x):
        return self.model.encode(x)

    def rollout(self, info_dict, action_candidates):
        self.seen.append({k: tuple(info_dict[k].shape) for k in self.keys if k in info_dict})
        return self.model.rollout(info_dict, action_candidates)


def run(policy, env, num_envs=2, episodes=4, **env_kwargs):
    world = sm.World(env, num_envs=num_envs, goal_conditioned=True, **env_kwargs)
    world.set_policy(policy)
    return world.evaluate(episodes=episodes, seed=0)


# ------------------------------------------------------------------------------------------- discrete: MultiGrid
# FindGoal ends when *every* agent reached the goal ('all'), so both agents must be planned for
OPEN_FINDGOAL = dict(agents=2, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1)


def grid_planner(mode, solver_cls=CategoricalCEMSolver, model=None, config=None, **solver_kwargs):
    cost = ShootingCostEvaluator(model or NavigationDynamics(), GoalMSE(per_agent=mode == 'per_agent'))
    kwargs = dict(batch_size=2, num_samples=128, n_steps=6, mode=mode, seed=0)
    kwargs.update(dict(topk=16) if solver_cls is CategoricalCEMSolver else dict(temperature=1.0))
    solver = solver_cls(cost, **{**kwargs, **solver_kwargs})
    return sm.WorldModelPolicy(solver, config or sm.PlanConfig(horizon=8, receding_horizon=2))


def run_grid(policy, env='MultiGrid-FindGoal-15x15-v0', num_envs=2, episodes=4, **env_kwargs):
    return run(policy, env, num_envs=num_envs, episodes=episodes, **(env_kwargs or OPEN_FINDGOAL))


@test
def discrete_planning_reaches_the_goal():
    for mode in PLANNING_MODES:
        for solver_cls in (CategoricalCEMSolver, CategoricalMPPISolver):
            res = run_grid(grid_planner(mode, solver_cls))
            assert res['success_rate'] == 100.0, (mode, solver_cls.__name__, res['episode_successes'])


@test
def planning_is_reproducible():
    a, b = run_grid(grid_planner('per_agent')), run_grid(grid_planner('per_agent'))
    assert np.array_equal(a['episode_lengths'], b['episode_lengths'])


@test
def plans_beat_random_actions():
    planned = run_grid(grid_planner('joint'), 'MultiGrid-Empty-8x8-v0', agents=2)
    world = sm.World('MultiGrid-Empty-8x8-v0', num_envs=2, agents=2)
    world.set_policy(sm.RandomPolicy(seed=0))
    random = world.evaluate(episodes=4, seed=0)
    assert planned['success_rate'] == 100.0 and planned['mean_length'] < random['mean_length']


@test
def joint_and_per_agent_costs_compose():
    cost = WeightedSum([(1.0, GoalMSE(per_agent=True)), (0.01, ControlPenalty(per_agent=True))])
    solver = CategoricalCEMSolver(ShootingCostEvaluator(NavigationDynamics(), cost), batch_size=2,
                                  num_samples=128, n_steps=6, topk=16, mode='per_agent')
    assert run_grid(sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2)))['success_rate'] == 100.0


@test
def mode_and_cost_must_agree():
    solver = CategoricalCEMSolver(ShootingCostEvaluator(NavigationDynamics(), GoalMSE()), num_samples=16,
                                  n_steps=1, topk=4, mode='per_agent')
    try:
        run_grid(sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=4, receding_horizon=1)), episodes=1)
        raise AssertionError("a joint cost must be rejected in per_agent mode")
    except ValueError as e:
        assert 'per-agent cost' in str(e)


@test
def discrete_history_and_action_blocks():
    """history_len=3, action_block=2: the model gets 1, 2 then 3 frames, and per-agent action blocks."""
    for mode in PLANNING_MODES:
        model = Recording(NavigationDynamics(action_block=2), ('pixels', 'action_history'))
        config = sm.PlanConfig(horizon=4, receding_horizon=1, history_len=3, action_block=2)
        res = run_grid(grid_planner(mode, model=model, config=config), num_envs=1, episodes=2)
        assert res['success_rate'] == 100.0, (mode, res['episode_successes'])
        frames = [s['pixels'][2] for s in model.seen]
        assert frames[0] == 1 and 2 in frames and max(frames) == 3, frames
        with_history = [s for s in model.seen if 'action_history' in s]
        assert with_history and all(s['action_history'][2:] == (s['pixels'][2] - 1, 2, 2) for s in with_history)
        assert all(s['pixels'][3:] == (2, 224, 224, 3) for s in model.seen)    # (agents, H, W, C)


# ------------------------------------------------------------------------------------------- continuous: point masses
class PointMassEnv(MultiAgentEnv):
    "Agents moving in [-1, 1]^2 with velocity actions, each to its own goal (terminates there)."
    metadata = {'render_modes': []}

    def __init__(self, agents: int = 2, max_steps: int = 60, speed: float = 0.1, goal_radius: float = 0.1):
        self.num_agents, self.max_steps, self.speed, self.goal_radius = agents, max_steps, speed, goal_radius
        box = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Dict({a: spaces.Dict({'position': box}) for a in range(agents)})
        self.action_space = spaces.Dict({a: spaces.Box(-1, 1, (2,), np.float32) for a in range(agents)})

    @property
    def state_space(self):
        return spaces.Box(-1, 1, (self.num_agents, 4), np.float32)

    def state(self):
        return np.concatenate([self.pos, self.goal], axis=-1)

    def _obs(self):
        return {a: {'position': self.pos[a].copy()} for a in range(self.num_agents)}

    def _infos(self):
        return {a: {'goal': self.goal[a].copy()} for a in range(self.num_agents)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        A = self.num_agents
        self.pos = self.np_random.uniform(-0.8, 0.8, (A, 2)).astype(np.float32)
        self.goal = self.np_random.uniform(-0.8, 0.8, (A, 2)).astype(np.float32)
        self.done, self.t = np.zeros(A, bool), 0
        return self._obs(), self._infos()

    def step(self, actions):
        for a in range(self.num_agents):
            if not self.done[a]:
                self.pos[a] = np.clip(self.pos[a] + self.speed * np.clip(actions[a], -1, 1), -1, 1)
        reached = ~self.done & (np.linalg.norm(self.pos - self.goal, axis=-1) < self.goal_radius)
        self.done |= reached
        self.t += 1
        A, trunc = range(self.num_agents), self.t >= self.max_steps
        return (self._obs(), {a: float(reached[a]) for a in A}, {a: bool(self.done[a]) for a in A},
                {a: trunc for a in A}, self._infos())


class PointMassDynamics(torch.nn.Module):
    "Exact, differentiable model of PointMassEnv (Dynamics protocol); embedding: positions."
    def __init__(self, speed: float = 0.1, action_block: int = 1):
        super().__init__()
        self.speed, self.action_block = speed, action_block

    def encode(self, x):
        x['emb'] = x['position'].float()
        return x

    def rollout(self, info_dict, action_candidates):
        pos = info_dict['position'][:, :, -1].float()                                   # (B, S, A, 2)
        actions = action_candidates.reshape(*action_candidates.shape[:-1], self.action_block, 2)
        trajectory = [pos]
        for t in range(actions.shape[2]):
            for j in range(self.action_block):
                pos = (pos + self.speed * actions[:, :, t, :, j].clamp(-1, 1)).clamp(-1, 1)
            trajectory.append(pos)
        info_dict['predicted_emb'] = torch.stack(trajectory, dim=2)
        return info_dict


sm.register('Test-PointMass-v0', PointMassEnv, family='test')
goal_as_position = partial(flat_goal_encode, goal_obs_key='position')   # the goal is a position, not an image


def point_planner(solver_cls, mode, model=None, config=None, **kwargs):
    cost = ShootingCostEvaluator(model or PointMassDynamics(), GoalMSE(per_agent=mode == 'per_agent'),
                                 encode_goal=goal_as_position)
    solver = solver_cls(cost, mode=mode, seed=0, **kwargs)
    return sm.WorldModelPolicy(solver, config or sm.PlanConfig(horizon=5, receding_horizon=1), history_keys=('position',))


CONTINUOUS_SOLVERS = {
    CEMSolver: dict(batch_size=2, num_samples=128, n_steps=5, topk=16),
    ICEMSolver: dict(batch_size=2, num_samples=128, n_steps=5, topk=16),
    MPPISolver: dict(batch_size=2, num_samples=128, n_steps=5, temperature=0.5),
    PredictiveSamplingSolver: dict(batch_size=2, num_samples=256),
    GradientSolver: dict(n_steps=15, num_samples=2, optimizer_cls=torch.optim.Adam, optimizer_kwargs={'lr': 0.3}),
    LagrangianSolver: dict(n_steps=8, n_outer_steps=2, num_samples=2, optimizer_kwargs={'lr': 0.3}),
}


@test
def continuous_planning_reaches_the_goals():
    for solver_cls, kwargs in CONTINUOUS_SOLVERS.items():
        for mode in PLANNING_MODES:
            res = run(point_planner(solver_cls, mode, **kwargs), 'Test-PointMass-v0', episodes=4)
            assert res['success_rate'] == 100.0, (solver_cls.__name__, mode, res['episode_successes'])


@test
def continuous_actions_are_recorded():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        world = sm.World('Test-PointMass-v0', num_envs=2, goal_conditioned=True)
        world.set_policy(point_planner(CEMSolver, 'joint', **CONTINUOUS_SOLVERS[CEMSolver]))
        world.collect(f'{tmp}/pm.h5', episodes=2, seed=0, progress=False)
        ds = sm.HDF5Dataset(path=f'{tmp}/pm.h5')
        actions = ds.get_col_data('action')
        assert actions.shape[1:] == (2, 2) and np.isnan(actions[ds.offsets + ds.lengths - 1]).all()
        assert np.isfinite(actions[ds.offsets]).all() and (np.abs(actions[np.isfinite(actions)]) <= 1).all()   # clipped


@test
def continuous_history_and_action_blocks():
    for mode in PLANNING_MODES:
        model = Recording(PointMassDynamics(action_block=2), ('position', 'action_history'))
        config = sm.PlanConfig(horizon=3, receding_horizon=1, history_len=3, action_block=2)
        res = run(point_planner(CEMSolver, mode, model=model, config=config, **CONTINUOUS_SOLVERS[CEMSolver]),
                  'Test-PointMass-v0', episodes=2)
        assert res['success_rate'] == 100.0, (mode, res['episode_successes'])
        frames = [s['position'][2] for s in model.seen]
        assert frames[0] == 1 and max(frames) == 3
        with_history = [s for s in model.seen if 'action_history' in s]
        assert with_history and all(s['action_history'][2:] == (s['position'][2] - 1, 2, 4) for s in with_history)


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
