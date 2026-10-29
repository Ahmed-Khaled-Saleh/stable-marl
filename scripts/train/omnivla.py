"""
Train OmniVLA-edge (stable_marl.wm.omnivla: the omni-modal navigation policy, torch only) on a FindGoal
dataset: one model shared by the agents, each acting on its own views and goal (pose and/or image).

    python scripts/data/collect_findgoal.py --config-name mixed   # MixedPolicy data (or any collecting policy)
    python scripts/train/omnivla.py
    python scripts/train/omnivla.py data.labels=collected data.hindsight_prob=1.0
    python scripts/train/omnivla.py pretrained=true model.encoding_size=1024   # from the released OmniVLA-edge weights

Reads the dataset <cache_dir>/datasets/<dataset_name> (its `state` column gives the expert's labels
for any collecting policy, see `OmniVLADataset`) and saves the model to
<cache_dir>/checkpoints/<output_model_name>/omnivla.pt, which scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, load_dataset
from stable_marl.wm.omnivla import OmniVLADataset, OmniVLAEdge, train_omnivla

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='omnivla')
def run(cfg):
    torch.manual_seed(cfg.seed)
    dataset = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir)
    meta = dataset.metadata
    num_actions = int(sm.make(meta['env_name'], **meta['env_kwargs']).action_space[0].n)
    model = OmniVLAEdge(num_actions=num_actions, **OmegaConf.to_container(cfg.model))
    if cfg.pretrained:
        loaded, kept = model.load_upstream(cache_dir=cfg.cache_dir)
        logger.info(f'released OmniVLA-edge weights: {len(loaded)} tensors loaded, {len(kept)} left as initialised')
    data = OmniVLADataset(dataset, context_size=cfg.model.context_size, horizon=cfg.model.horizon,
                          **OmegaConf.to_container(cfg.data))
    logger.info(f"{cfg.dataset_name}: {len(dataset.lengths)} episodes, {len(data)} samples (agent steps), "
                f"{num_actions} actions, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    train_omnivla(model, data, **OmegaConf.to_container(cfg.train),
                  log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}))
    path = model.save(Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'omnivla.pt')
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
