"""
Collect a RoboFactory dataset of motion-planning expert (and random) episodes.

    python scripts/data/collect_robofactory.py                                   # LiftBarrier, 150 expert episodes
    python scripts/data/collect_robofactory.py env.task=TakePhoto env.camera_size=[64,64] expert_episodes=300
    python scripts/data/collect_robofactory.py shard=3                           # one job of an array: own seeds and file

Writes <cache_dir>/datasets/<dataset_name>(_shard<k>)(.h5), readable with
`stable_marl.data.load_dataset(...)`; shards are merged with `stable_marl.data.merge`.

The expert (RoboFactory's motion planner, mplib 0.1.1) needs numpy<2: run this in an environment with
`pip install 'stable-marl[robofactory]' 'numpy<2'`. The assets are downloaded on first use; on compute
nodes without internet, run `python -c "from stable_marl.envs.robofactory import download_assets; download_assets()"`
beforehand on a login node.
"""
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, get_format
from stable_marl.envs.robofactory import RoboFactoryExpert

logger = logging.getLogger(__name__)


class SuccessOnly:
    "A dataset writer keeping only the episodes that ended with the task done (every agent terminated)."
    def __init__(self, writer):
        self.writer, self.kept, self.dropped = writer, 0, 0

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def __enter__(self):
        self.writer.__enter__()
        return self

    def __exit__(self, *exc):
        return self.writer.__exit__(*exc)

    def write_episodes(self, episodes):
        def successful():
            for ep in episodes:
                done = bool(np.asarray(ep['terminated'][-1]).all())
                self.kept, self.dropped = self.kept + done, self.dropped + (not done)
                if done:
                    yield ep
        self.writer.write_episodes(successful())


@hydra.main(version_base=None, config_path='./config', config_name='robofactory')
def run(cfg):
    env = OmegaConf.to_container(cfg.env)
    task = env.pop('task')
    world = sm.World(f'RoboFactory-{task}-v0', **cfg.world, **{k: v for k, v in env.items() if v is not None})
    suffix = '.h5' if cfg.format == 'hdf5' else ''
    name = cfg.dataset_name or f'robofactory_{task.lower()}'
    if cfg.shard is not None:
        name = f'{name}_shard{cfg.shard}'
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='datasets')) / f'{name}{suffix}'
    seed = cfg.seed + 10_000_000 * (cfg.shard or 0)

    written = False
    for name_, policy, episodes in (('expert', RoboFactoryExpert(), cfg.expert_episodes),
                                    ('random', sm.RandomPolicy(seed=seed), cfg.random_episodes)):
        if not episodes:
            continue
        world.set_policy(policy)
        writer = get_format(cfg.format).open_writer(path, mode='append' if written else 'overwrite')
        if name_ == 'expert' and cfg.only_success:
            writer = SuccessOnly(writer)
        world.collect(writer=writer, episodes=episodes, seed=seed + (1_000_000 if name_ == 'random' else 0))
        written = True
        kept = f' ({writer.kept} kept, {writer.dropped} unsuccessful dropped)' if isinstance(writer, SuccessOnly) else ''
        logger.info(f'{episodes} {name_} episodes{kept} -> {path}')

    ds = sm.data.load_dataset(path)
    logger.info(f'{path}: {len(ds.lengths)} episodes, {int(ds.lengths.sum())} steps, columns {ds.column_names}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
