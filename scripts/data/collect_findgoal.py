"""
Collect a FindGoal dataset of expert and random episodes (stable-worldmodel's data scripts).

    python scripts/data/collect_findgoal.py
    python scripts/data/collect_findgoal.py dataset_name=findgoal_global env.obs_mode=global format=folder

Writes <cache_dir>/datasets/<dataset_name>(.h5), readable with
`stable_marl.data.load_dataset('<dataset_name>')`.
"""
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, get_format
from stable_marl.envs.multigrid import GoToGoalPolicy

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    env = OmegaConf.to_container(cfg.env)
    world = sm.World(env.pop('name'), **cfg.world, goal_conditioned=True, **env)
    suffix = '.h5' if cfg.format == 'hdf5' else ''
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='datasets')) / f'{cfg.dataset_name}{suffix}'
    options = OmegaConf.to_container(cfg.options) if cfg.options else None

    for i, (name, policy, episodes) in enumerate((('expert', GoToGoalPolicy(), cfg.expert_episodes),
                                                   ('random', sm.RandomPolicy(seed=cfg.seed), cfg.random_episodes))):
        if not episodes:
            continue
        world.set_policy(policy)
        writer = get_format(cfg.format).open_writer(path, mode='overwrite' if i == 0 else 'append')
        world.collect(writer=writer, episodes=episodes, seed=cfg.seed + 1_000_000 * i, options=options)
        logger.info(f'{episodes} {name} episodes -> {path}')

    ds = sm.data.load_dataset(path)
    logger.info(f'{path}: {len(ds.lengths)} episodes, {int(ds.lengths.sum())} steps, columns {ds.column_names}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
