"""
Train MAMBA (Egorov & Shpilman, AAMAS 2022) online on a MultiGrid env: with its Flatland-style
shaped reward (the paper's method), or goal-conditioned with hindsight relabelling (GC-MAMBA).

    python scripts/train/mamba.py
    python scripts/train/mamba.py goal_conditioned=true
    python scripts/train/mamba.py train.episodes=5000 +mamba_config.DEVICE=cuda

Saves <cache_dir>/checkpoints/<output_model_name>/mamba.pt (with the env it was trained on), which
scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir
from stable_marl.wm.mamba import Mamba

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='mamba')
def run(cfg):
    env = OmegaConf.to_container(cfg.env)
    env_name = env.pop('name')
    world = sm.World(env_name, num_envs=1, goal_conditioned=True, **env)
    num_actions = int(world.envs.envs[0].unwrapped.action_space[0].n)
    model = Mamba(env['agents'], num_actions, goal_conditioned=cfg.goal_conditioned, obs_keys=list(cfg.obs_keys),
                  grid_size=env['size'], finish_value=cfg.rewards.finish_value, near_coeff=cfg.rewards.near_coeff,
                  relabel_k=cfg.relabel_k, seed=cfg.seed, **OmegaConf.to_container(cfg.mamba_config))
    logger.info(f"{'GC-MAMBA' if cfg.goal_conditioned else 'MAMBA'} on {env_name} {env}")
    model.train(world, episodes=cfg.train.episodes, seed=cfg.seed, log_every=cfg.train.log_every,
                log=lambda row: logger.info({k: round(v, 3) for k, v in row.items() if not k.startswith('mamba/')}))
    name = cfg.output_model_name or ('gc_mamba_findgoal' if cfg.goal_conditioned else 'mamba_findgoal')
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / name / 'mamba.pt'
    model.save(path, env_name=env_name, env_kwargs=env)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
