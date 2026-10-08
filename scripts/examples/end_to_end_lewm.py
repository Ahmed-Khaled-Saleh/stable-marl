"""
Decentralized LeWMs on a small FindGoal (random goals and starts): collect expert + random data,
train one LeWM per agent on its own view, then let each agent plan with its own model.

    python scripts/examples/end_to_end_lewm.py
    python scripts/examples/end_to_end_lewm.py inputs=[pixels,position,direction] env.obs_mode=global
    python scripts/examples/end_to_end_lewm.py env.agents=1 model.image_size=64 train.epochs=40

Config: scripts/examples/config/end_to_end_lewm.yaml. Extra inputs (position, direction) are read
from the infos next to the image (with the goal's position for the goal). The model is evaluated
with the goal scored at the last predicted step ('last', as in stable-worldmodel) and at the plan's
best step ('min'), next to the random and expert policies. The same steps, run separately with
saved datasets and checkpoints: scripts/data, scripts/train, scripts/plan.
"""
import tempfile
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_marl.wm import DecentralizedWorldModel, train_world_model


@hydra.main(version_base=None, config_path='./config', config_name='end_to_end_lewm')
def run(cfg):
    torch.manual_seed(cfg.seed)
    env = OmegaConf.to_container(cfg.env)
    env_name = env.pop('name')
    inputs = list(cfg.inputs)
    num_agents, num_actions = env['agents'], int(sm.make(env_name, **env).action_space[0].n)

    # 1. data: expert and random episodes
    tmp, t = tempfile.mkdtemp(), time.time()
    for policy, episodes, seed in ((GoToGoalPolicy(), cfg.data.expert_episodes, cfg.seed),
                                   (sm.RandomPolicy(seed=cfg.seed), cfg.data.random_episodes, cfg.seed + 50_000)):
        world = sm.World(env_name, num_envs=cfg.data.num_envs, max_episode_steps=cfg.data.max_episode_steps,
                         goal_conditioned=True, **env)
        world.set_policy(policy)
        world.collect(f'{tmp}/data.h5', episodes=episodes, seed=seed, progress=False)
    ds = sm.HDF5Dataset(path=f'{tmp}/data.h5', num_steps=cfg.model.history_size + 1,
                        keys_to_load=[*inputs, 'action', 'terminated'])
    print(f"agents={num_agents} obs_mode={env.get('obs_mode')} image_size={cfg.model.image_size} inputs={inputs}: "
          f"{len(ds.lengths)} episodes, {len(ds)} clips ({time.time() - t:.0f}s)")

    # 2. one LeWM per agent
    extra = {k: int(np.prod(ds.get_col_data(k).shape[2:])) or 1 for k in inputs if k != 'pixels'}
    model = DecentralizedWorldModel.build_lewm(num_agents=num_agents, num_actions=num_actions,
                                               **({'extra_inputs': extra} if extra else {}),
                                               **OmegaConf.to_container(cfg.model))
    t = time.time()
    hist = train_world_model(model, ds, epochs=cfg.train.epochs, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                             sigreg_kwargs={'num_proj': cfg.train.sigreg_num_proj}, log=None)
    print(f"trained {cfg.train.epochs} epochs in {time.time() - t:.0f}s: train {hist['train_loss'][0]:.3f} -> "
          f"{hist['train_loss'][-1]:.3f}, val {hist['val_loss'][0]:.3f} -> {hist['val_loss'][-1]:.3f}")

    # 3. each agent plans with its own model
    p = cfg.planner
    def planner(step_reduction):
        cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True, step_reduction=step_reduction))
        solver = CategoricalCEMSolver(cost, batch_size=cfg.eval.num_envs, num_samples=p.num_samples, n_steps=p.n_steps,
                                      topk=p.topk, mode='per_agent', seed=cfg.seed)
        config = sm.PlanConfig(horizon=p.horizon, receding_horizon=p.receding_horizon, history_len=p.history_len)
        return sm.WorldModelPolicy(solver, config, history_keys=inputs)

    policies = [('random', sm.RandomPolicy(seed=cfg.seed + 1))]
    policies += [(f'LeWM, goal at the {"last" if r == "last" else "best"} step', planner(r)) for r in p.step_reductions]
    policies += [('expert', GoToGoalPolicy())]
    for name, policy in policies:
        t = time.time()
        world = sm.World(env_name, num_envs=cfg.eval.num_envs, max_episode_steps=cfg.eval.max_episode_steps,
                         goal_conditioned=True, **env)
        world.set_policy(policy)
        r = world.evaluate(episodes=cfg.eval.episodes, seed=cfg.eval.seed)
        print(f"{name:28s} success {r['success_rate']:5.1f}%  mean length {r['mean_length']:5.1f}  ({time.time() - t:.0f}s)")


if __name__ == '__main__':
    run()
