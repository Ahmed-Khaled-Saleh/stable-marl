"""
How do policies cope with environment changes they never saw in training? Train on one setting of
FindGoal (15 x 15 grid, 7 x 7 local views, 6 inner walls), then evaluate the same policies under shifts:
other colors, farther goals, more inner walls, noisy or occluded views.

    python scripts/examples/generalization.py
    python scripts/examples/generalization.py models=[gcbc,hiql] eval.episodes=50

Config: scripts/examples/config/generalization.yaml. A shift changes env arguments (`env_kwargs`,
e.g. the layout), factors of variation (reset `options`) or the agents' views (visual `wrappers`, which
leave the goal images clean). The same settings evaluate saved checkpoints: `eval.env_kwargs`,
`eval.options` and `eval.wrappers` of scripts/plan/eval_wm.py and scripts/plan/eval_ff.py.
"""
import tempfile
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy


def collect(cfg, env_name, env, path, random_episodes):
    for policy, episodes, seed in ((GoToGoalPolicy(), cfg.data.expert_episodes, cfg.seed),
                                   (sm.RandomPolicy(seed=cfg.seed), random_episodes, cfg.seed + 50_000)):
        if not episodes:
            continue
        world = sm.World(env_name, num_envs=cfg.data.num_envs, max_episode_steps=cfg.data.max_episode_steps,
                         goal_conditioned=True, **env)
        world.set_policy(policy)
        world.collect(path, episodes=episodes, seed=seed, progress=False)


def lewm_policy(cfg, env_name, env, num_actions, tmp):
    from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
    from stable_marl.wm import DecentralizedWorldModel, train_world_model
    torch.manual_seed(cfg.seed)
    collect(cfg, env_name, env, f'{tmp}/lewm.h5', cfg.data.random_episodes)
    c = cfg.lewm
    ds = sm.HDF5Dataset(path=f'{tmp}/lewm.h5', num_steps=c.model.history_size + 1, keys_to_load=['pixels', 'action', 'terminated'])
    model = DecentralizedWorldModel.build_lewm(num_agents=env['agents'], num_actions=num_actions, **OmegaConf.to_container(c.model))
    train_world_model(model, ds, epochs=c.train.epochs, batch_size=c.train.batch_size, lr=c.train.lr,
                      sigreg_kwargs={'num_proj': c.train.sigreg_num_proj}, log=None)
    p = c.planner
    cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True, step_reduction=p.step_reduction))
    solver = CategoricalCEMSolver(cost, batch_size=cfg.eval.num_envs, num_samples=p.num_samples, n_steps=p.n_steps,
                                  topk=p.topk, mode='per_agent', seed=cfg.seed)
    config = sm.PlanConfig(horizon=p.horizon, receding_horizon=p.receding_horizon, history_len=p.history_len)
    return lambda: sm.WorldModelPolicy(solver, config, history_keys=['pixels'])


def gcrl_policy(cfg, name, env_name, env, num_actions, tmp):
    from stable_marl.wm.gcrl import IndependentGCRL, agent_datasets
    collect(cfg, env_name, env, f'{tmp}/expert.h5', 0)
    data = agent_datasets(sm.HDF5Dataset(path=f'{tmp}/expert.h5'), list(cfg.gcrl.obs_keys))
    model = IndependentGCRL.create(name, data, num_actions, obs_keys=list(cfg.gcrl.obs_keys), seed=cfg.seed,
                                   **OmegaConf.to_container(cfg.gcrl.agent_config))
    model.train(data, steps=cfg.gcrl.steps, seed=cfg.seed)
    return lambda: sm.FeedForwardPolicy(model)


def evaluate(cfg, env_name, env, make_policy, shift):
    world = sm.World(env_name, num_envs=cfg.eval.num_envs, max_episode_steps=cfg.eval.max_episode_steps, goal_conditioned=True,
                     extra_wrappers=[hydra.utils.instantiate(w) for w in shift.get('wrappers', [])] or None,
                     **{**env, **shift.get('env_kwargs', {})})
    world.set_policy(make_policy())
    options = OmegaConf.to_container(shift['options']) if shift.get('options') else None
    return world.evaluate(episodes=cfg.eval.episodes, seed=cfg.eval.seed, options=options)


@hydra.main(version_base=None, config_path='./config', config_name='generalization')
def run(cfg):
    env = OmegaConf.to_container(cfg.env)
    env_name = env.pop('name')
    num_actions = int(sm.make(env_name, **env).action_space[0].n)
    tmp = tempfile.mkdtemp()

    policies = {'expert': GoToGoalPolicy, 'random': lambda: sm.RandomPolicy(seed=cfg.seed + 1)}
    for name in cfg.models:
        t = time.time()
        policies[name] = (lewm_policy(cfg, env_name, env, num_actions, tmp) if name == 'lewm'
                          else gcrl_policy(cfg, name, env_name, env, num_actions, tmp))
        print(f'trained {name} ({time.time() - t:.0f}s)', flush=True)

    names = [*cfg.models, 'expert', 'random']
    rows = {}
    for shift_name, shift in cfg.shifts.items():
        t = time.time()
        rows[shift_name] = {n: evaluate(cfg, env_name, env, policies[n], shift)['success_rate'] for n in names}
        print(f"{shift_name:24s} " + '  '.join(f'{n} {v:5.1f}%' for n, v in rows[shift_name].items())
              + f'  ({time.time() - t:.0f}s)', flush=True)

    print(f"\nsuccess rate (%), {cfg.eval.episodes} episodes per setting\n")
    print('| setting | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
    for shift_name, row in rows.items():
        print(f'| {shift_name} | ' + ' | '.join(f'{row[n]:.0f}' for n in names) + ' |')


if __name__ == '__main__':
    run()
