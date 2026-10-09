"""
Evaluate a feed-forward policy, i.e. one forward pass per step without planning (stable-worldmodel's
plan/eval_ff.py): the goal-conditioned baselines of scripts/train/gcrl.py (`gcrl.pkl` checkpoints) and
MAMBA of scripts/train/mamba.py (`mamba.pt`, evaluated on the env it was trained on).

    python scripts/plan/eval_ff.py                                # GCBC
    python scripts/plan/eval_ff.py policy=hiql_findgoal temperature=0.3
    python scripts/plan/eval_ff.py policy=gc_mamba_findgoal

Each agent acts on its own view (or features) and its own goal. The results (and the config) are
appended to a text file next to the checkpoint, and a few episodes are saved as videos.
"""
import logging
import time
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, load_dataset

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='eval_ff')
def run(cfg):
    checkpoint = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.policy
    if (checkpoint / 'mamba.pt').exists():                       # MAMBA: online, its checkpoint holds its env
        from stable_marl.wm.mamba import Mamba
        model = Mamba.load(checkpoint / 'mamba.pt')
        env_name, env_kwargs = model.checkpoint['env_name'], model.checkpoint['env_kwargs']
        policy = model.policy(deterministic=cfg.temperature == 0)
    else:                                                        # goal-conditioned baselines: the dataset's env
        from stable_marl.wm.gcrl import IndependentGCRL
        meta = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir).metadata
        env_name, env_kwargs = meta['env_name'], meta['env_kwargs']
        model = IndependentGCRL.load(checkpoint / 'gcrl.pkl', seed=cfg.seed)
        model.temperature = cfg.temperature
        policy = sm.FeedForwardPolicy(model)
    world = sm.World(env_name, num_envs=cfg.eval.num_envs, goal_conditioned=True,
                     extra_wrappers=[hydra.utils.instantiate(w) for w in cfg.eval.wrappers] or None,
                     **{**env_kwargs, **cfg.eval.env_kwargs, 'max_steps': cfg.eval.max_episode_steps})
    options = OmegaConf.to_container(cfg.eval.options) if cfg.eval.options else None
    world.set_policy(policy)

    start = time.time()
    results = world.evaluate(episodes=cfg.eval.episodes, seed=cfg.eval.seed, options=options)
    elapsed = time.time() - start
    metrics = {k: results[k] for k in ('success_rate', 'mean_length', 'mean_return')}
    logger.info(f'{cfg.policy}: {metrics} ({elapsed:.0f}s)')

    out_dir = Path(cfg.output.dir) if cfg.output.dir else checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / cfg.output.filename).open('a') as f:
        f.write(f"\n==== CONFIG ====\n{OmegaConf.to_yaml(cfg, resolve=True)}\n==== RESULTS ====\n"
                f"metrics: {metrics}\nepisode_successes: {results['episode_successes'].tolist()}\n"
                f"evaluation_time: {elapsed:.1f} seconds\n")
    logger.info(f'results appended to {out_dir / cfg.output.filename}')

    if cfg.eval.video_episodes:
        video_dir = out_dir / f'{cfg.policy}_videos'
        world.evaluate(episodes=cfg.eval.video_episodes, seed=cfg.eval.seed, options=options, video=video_dir)
        logger.info(f'videos in {video_dir}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
