"""
Test the external wrappers (stable_marl/wrappers/external.py) with each library's own checks:

  * PettingZoo: `parallel_api_test`, plus agents terminating at different times
  * TorchRL:    `check_env_specs` and rollouts through `torchrl.envs.libs.pettingzoo.PettingZooWrapper`
  * RLlib:      `check_multiagent_environments`, early termination, and a PPO training iteration

"Early termination" uses a 5x5 Empty env with success_termination_mode='all', where
one agent reaches the goal while the other keeps acting.

Run:  python tests/test_external_wrappers.py [--skip-rllib-training]
"""
import argparse
import contextlib
import io
import logging
import warnings

import gymnasium as gym
import numpy as np

import stable_marl.envs
from stable_marl.envs.multigrid.core.actions import Action
from stable_marl.envs.multigrid import EmptyEnv, FindGoalEnv, RedBlueDoorsEnv
from stable_marl.wrappers.external import (
    PettingZooWrapper, RLlibWrapper, TorchRLPettingZooWrapper,
    register_rllib_envs, to_pettingzoo_env, to_rllib_env,
)

warnings.filterwarnings('ignore')

ENVS = {
    'FindGoal': lambda: FindGoalEnv(agents=2, num_obstacles=6),
    'RedBlueDoors': lambda: RedBlueDoorsEnv(agents=2),
    'Empty5x5-early-termination': lambda: EmptyEnv(size=5, agents=2, success_termination_mode='all'),
    'gym.make Empty': lambda: gym.make('MultiGrid-Empty-8x8-v0', agents=2),
}

# Agent 0 walks (1,1) -> (3,1) -> turns down -> (3,3) = goal of the 5x5 Empty env; agent 1 only turns
F, R, L = int(Action.forward), int(Action.right), int(Action.left)
GOAL_PLAN = [F, F, R, F, F]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def put_agent0_before_goal(base_env):
    """Place agent 0 of a FindGoal env right in front of the goal, facing it."""
    from stable_marl.envs.multigrid.core.constants import Direction
    gx, gy = map(int, base_env.goal_pos)
    for d in Direction:
        dx, dy = d.to_vec()
        if base_env.grid.get(gx - dx, gy - dy) is None:
            base_env.agents[0].state.pos, base_env.agents[0].state.dir = (gx - dx, gy - dy), d
            return
    raise RuntimeError("no free cell next to the goal")


def check_findgoal_finish(wrapper, ids):
    """FindGoal: the observation of an agent reaching the goal must stay inside the observation space."""
    a0, a1 = ids
    wrapper.reset(seed=0)
    put_agent0_before_goal(wrapper.env.unwrapped)
    obs, rew, term, trunc, info = wrapper.step({a0: F, a1: L})
    assert term[a0] and rew[a0] > 0
    assert wrapper.observation_space(a0).contains(obs[a0]) if callable(wrapper.observation_space) \
        else True
    return "FindGoal agent reaching the goal OK"


def check_early_termination(wrapper, ids):
    """Agent 0 finishes first: it must be reported in that step, then dropped everywhere."""
    a0, a1 = ids
    wrapper.reset(seed=0)
    for k, action in enumerate(GOAL_PLAN):
        obs, rew, term, trunc, info = wrapper.step({a0: action, a1: L})
    assert term[a0] and not term[a1] and rew[a0] > 0, f"agent 0 should have just finished: {term}, {rew}"
    assert a0 in obs and a1 in obs
    assert list(wrapper.agents) == [a1], f"agents after finish: {wrapper.agents}"
    for _ in range(3):
        obs, rew, term, trunc, info = wrapper.step({a0: F, a1: L})   # a0's action must be ignored
        for d in (obs, rew, term, trunc, info):
            assert a0 not in d, f"finished agent still reported: {d.keys()}"
        assert a1 in obs
    return "early termination OK"


# ------------------------------------------------------------------ PettingZoo
def test_pettingzoo():
    from pettingzoo.test import parallel_api_test
    print("PettingZoo")
    for name, make in ENVS.items():
        quiet(parallel_api_test, PettingZooWrapper(make()), num_cycles=500)
        print(f"  {name:28s} parallel_api_test OK")
    PZEnv = to_pettingzoo_env(EmptyEnv, metadata={'name': 'empty_v0'})
    quiet(parallel_api_test, PZEnv(agents=2), num_cycles=100)
    print(f"  {'to_pettingzoo_env(EmptyEnv)':28s} parallel_api_test OK")
    print(f"  {'Empty5x5':28s}", check_early_termination(PettingZooWrapper(ENVS['Empty5x5-early-termination']()), (0, 1)))
    print(f"  {'FindGoal':28s}", check_findgoal_finish(PettingZooWrapper(ENVS['FindGoal']()), (0, 1)))


# ------------------------------------------------------------------ TorchRL
def test_torchrl():
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper as TorchRLPettingZoo
    import torch
    from torchrl.envs.utils import check_env_specs
    logging.getLogger('torchrl').setLevel(logging.WARNING)
    print("TorchRL")
    for name, make in ENVS.items():
        env = TorchRLPettingZoo(env=TorchRLPettingZooWrapper(make()), use_mask=True)
        check_env_specs(env)
        env.set_seed(0)
        steps = 0
        for _ in range(5):   # several episodes, so agents finish at different times
            td = env.rollout(300, break_when_any_done=False)
            steps += td.batch_size[0]
        print(f"  {name:28s} check_env_specs OK | 5 rollouts ({steps} steps) OK | group_map {env.group_map}")
    print(f"  {'Empty5x5 (agent ids)':28s}",
          check_early_termination(TorchRLPettingZooWrapper(ENVS['Empty5x5-early-termination']()), ('agent_0', 'agent_1')))
    env = TorchRLPettingZoo(env=TorchRLPettingZooWrapper(ENVS['FindGoal']()), use_mask=True)
    env.set_seed(0)
    td = env.reset()
    put_agent0_before_goal(env._env.env.unwrapped)
    td = env.rand_action(td)   # action layout from the spec (categorical or one-hot)
    actions = torch.tensor([F, L])
    if td['agent', 'action'].dim() == 2:   # one-hot
        actions = torch.nn.functional.one_hot(actions, td['agent', 'action'].shape[-1])
    td['agent', 'action'] = actions.to(td['agent', 'action'].dtype)
    td = env.step(td)
    assert td['next', 'agent', 'terminated'][0].item() and td['next', 'agent', 'reward'][0].item() > 0
    print(f"  {'FindGoal':28s} TorchRL step with an agent reaching the goal OK")


# ------------------------------------------------------------------ RLlib
def test_rllib(train: bool):
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments
    print("RLlib")
    for name, make in ENVS.items():
        env = RLlibWrapper(make())
        check_multiagent_environments(env)
        obs, _ = env.reset(seed=0)
        assert all(env.observation_spaces[i].contains(o) for i, o in obs.items())
        for _ in range(200):
            obs, rew, term, trunc, info = env.step({i: env.action_spaces[i].sample() for i in env.agents})
            assert all(env.observation_spaces[i].contains(o) for i, o in obs.items())
            if term['__all__'] or trunc['__all__']:
                obs, _ = env.reset()
        print(f"  {name:28s} check_multiagent_environments OK | obs space {env.observation_spaces[0].shape}")
    print(f"  {'Empty5x5':28s}", check_early_termination(RLlibWrapper(ENVS['Empty5x5-early-termination']()), (0, 1)))
    env = RLlibWrapper(ENVS['FindGoal']())
    env.reset(seed=0)
    put_agent0_before_goal(env.env.unwrapped)
    obs, rew, term, trunc, info = env.step({0: F, 1: L})
    assert term[0] and env.observation_spaces[0].contains(obs[0])
    print(f"  {'FindGoal':28s} agent reaching the goal OK")

    pov = RLlibWrapper(RedBlueDoorsEnv(agents=2), obs_keys=('pixels',), flatten=False)
    obs, _ = pov.reset(seed=0)
    assert pov.observation_spaces[0].contains(obs[0])
    print(f"  {'obs_keys=(pixels,) flatten=False':28s} obs space {pov.observation_spaces[0].shape} OK")

    if not train:
        return
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    register_rllib_envs()
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, logging_level=logging.ERROR)
    try:
        for env_name, env_config in [('MultiGrid-Empty-5x5-v0', {'agents': 2, 'success_termination_mode': 'all'}),
                                     ('MultiGrid-RedBlueDoors-6x6-v0', {'agents': 2})]:
            config = (PPOConfig()
                      .environment(env_name, env_config=env_config)
                      .env_runners(num_env_runners=0)
                      .multi_agent(policies={'shared'}, policy_mapping_fn=lambda aid, *a, **k: 'shared')
                      .training(train_batch_size=512, minibatch_size=128, num_epochs=1))
            algo = config.build_algo()
            for _ in range(2):
                result = algo.train()
            algo.stop()
            steps = result.get('num_env_steps_sampled_lifetime')
            print(f"  {env_name:28s} PPO (shared policy) 2 training iterations OK ({steps} env steps)")
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-rllib-training', action='store_true')
    args = parser.parse_args()
    test_pettingzoo()
    test_torchrl()
    test_rllib(train=not args.skip_rllib_training)
    print("\nAll external wrapper tests passed.")
