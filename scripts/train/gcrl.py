"""
Train MangoBench's decentralized goal-conditioned baselines: one agent per agent, each on its own
trajectories, with OGBench's JAX implementations (needs `pip install 'stable-marl[gcrl]'`).

    python scripts/train/gcrl.py                                  # GCBC on the agents' views
    python scripts/train/gcrl.py agent=hiql
    python scripts/train/gcrl.py agent=crl agent_config.alpha=0.1 train.steps=100000

Reads the dataset <cache_dir>/datasets/<dataset_name> (its metadata gives the env) and saves the agents
to <cache_dir>/checkpoints/<output_model_name>/gcrl.pkl, which scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, load_dataset
from stable_marl.wm.gcrl import IndependentGCRL, agent_datasets

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='gcrl')
def run(cfg):
    obs_keys = list(cfg.obs_keys)
    dataset = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir)
    meta = dataset.metadata
    num_actions = int(sm.make(meta['env_name'], **meta['env_kwargs']).action_space[0].n)
    data = agent_datasets(dataset, obs_keys)
    logger.info(f"{cfg.dataset_name}: {len(dataset.lengths)} episodes, {len(data)} agents, {num_actions} actions, "
                f"observations {obs_keys}, transitions per agent {[int(d['valids'].sum()) for d in data]}")

    model = IndependentGCRL.create(cfg.agent, data, num_actions, obs_keys=obs_keys, seed=cfg.seed,
                                   **OmegaConf.to_container(cfg.agent_config))
    model.train(data, steps=cfg.train.steps, log_every=cfg.train.log_every, seed=cfg.seed,
                log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}))
    path = model.save(Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'gcrl.pkl')
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
