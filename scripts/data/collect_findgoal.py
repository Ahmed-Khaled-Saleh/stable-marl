"""
Collect a FindGoal dataset with one or more policies (stable-worldmodel's data scripts): by default
expert and random episodes, or any policies listed in `policies`.

    python scripts/data/collect_findgoal.py
    python scripts/data/collect_findgoal.py dataset_name=findgoal_global env.obs_mode=global format=folder
    python scripts/data/collect_findgoal.py --config-name mixed      # MixedPolicy episodes -> findgoal_mixed
    python scripts/data/collect_findgoal.py 'policies=[{_target_: stable_marl.envs.multigrid.MixedPolicy, episodes: 450, epsilon: 0.5}]'

Writes <cache_dir>/datasets/<dataset_name>(.h5), readable with
`stable_marl.data.load_dataset('<dataset_name>')`.
"""
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, get_format

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    env = OmegaConf.to_container(cfg.env)
    world = sm.World(env.pop('name'), **cfg.world, goal_conditioned=True, **env)
    suffix = '.h5' if cfg.format == 'hdf5' else ''
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='datasets')) / f'{cfg.dataset_name}{suffix}'
    options = OmegaConf.to_container(cfg.options) if cfg.options else None

    written = 0
    for i, spec in enumerate(OmegaConf.to_container(cfg.policies, resolve=True)):
        episodes = spec.pop('episodes')
        if not episodes:
            continue
        policy = hydra.utils.instantiate(spec)
        world.set_policy(policy)
        writer = get_format(cfg.format).open_writer(path, mode='overwrite' if written == 0 else 'append')
        world.collect(writer=writer, episodes=episodes, seed=cfg.seed + 1_000_000 * i, options=options)
        written += 1
        logger.info(f'{episodes} {type(policy).__name__} episodes -> {path}')

    ds = sm.data.load_dataset(path)
    logger.info(f'{path}: {len(ds.lengths)} episodes, {int(ds.lengths.sum())} steps, columns {ds.column_names}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
