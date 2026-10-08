"""
Evaluate a world model by planning with it (stable-worldmodel's plan/eval_wm.py, for several agents).

    python scripts/plan/eval_wm.py                                   # the LeWM of scripts/train/lewm.py
    python scripts/plan/eval_wm.py policy=random
    python scripts/plan/eval_wm.py solver=categorical_mppi mode=joint
    python scripts/plan/eval_wm.py policy=lewm_findgoal_pos inputs=[pixels,position,direction]

Each agent plans with its model (its own view, its own goal image): the solver optimizes action
sequences scored by the objective, and `WorldModelPolicy` executes them with a receding horizon. The
results (and the config) are appended to a text file, and a few episodes are saved as panel videos
(each agent's view next to its goal).
"""
import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import ReplayBuffer, get_cache_dir, load_dataset
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.planning import ShootingCostEvaluator
from stable_marl.plot import save_panel_videos
from stable_marl.wm import DecentralizedWorldModel

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver('eq', lambda a, b: a == b, replace=True)


def make_policy(cfg):
    if cfg.policy == 'random':
        return sm.RandomPolicy(seed=cfg.seed), None
    if cfg.policy == 'expert':
        return GoToGoalPolicy(), None
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / f'{cfg.policy}.pt'
    model = DecentralizedWorldModel.load(path)
    cost = ShootingCostEvaluator(model, hydra.utils.instantiate(cfg.objective))
    solver = hydra.utils.instantiate(cfg.solver, cost=cost)
    config = sm.PlanConfig(**cfg.plan_config)
    return sm.WorldModelPolicy(solver, config, history_keys=tuple(cfg.inputs)), path


@hydra.main(version_base=None, config_path='./config', config_name='findgoal')
def run(cfg):
    meta = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir).metadata     # the env of the training data
    world = sm.World(meta['env_name'], num_envs=cfg.eval.num_envs, goal_conditioned=True,
                     **{**meta['env_kwargs'], 'max_steps': cfg.eval.max_episode_steps})
    policy, checkpoint = make_policy(cfg)
    world.set_policy(policy)

    start = time.time()
    results = world.evaluate(episodes=cfg.eval.episodes, seed=cfg.eval.seed)
    elapsed = time.time() - start
    metrics = {k: results[k] for k in ('success_rate', 'mean_length', 'mean_return')}
    logger.info(f'{cfg.policy}: {metrics} ({elapsed:.0f}s)')

    out_dir = Path(cfg.output.dir) if cfg.output.dir else \
        (checkpoint.parent if checkpoint else Path(__file__).parent / 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / cfg.output.filename).open('a') as f:
        f.write(f"\n==== CONFIG ====\n{OmegaConf.to_yaml(cfg, resolve=True)}\n==== RESULTS ====\n"
                f"metrics: {metrics}\nepisode_successes: {results['episode_successes'].tolist()}\n"
                f"evaluation_time: {elapsed:.1f} seconds\n")
    logger.info(f'results appended to {out_dir / cfg.output.filename}')

    if cfg.eval.video_episodes:
        world.set_policy(policy)
        buffer = ReplayBuffer(max_steps=cfg.eval.video_episodes * (cfg.eval.max_episode_steps + 1))
        world.collect(writer=buffer, episodes=cfg.eval.video_episodes, seed=cfg.eval.seed, progress=False)
        episodes = list(buffer.episodes())
        panels = {'views': [np.stack(ep['pixels']) for ep in episodes],     # (T, agents, H, W, 3)
                  'goals': [np.stack(ep['goal']) for ep in episodes]}
        video_dir = out_dir / f'{cfg.policy}_videos'
        save_panel_videos(video_dir, panels, fps=4)
        logger.info(f'videos in {video_dir}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
