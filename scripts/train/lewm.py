"""
Train one LeWM world model per agent, each on its own view (stable-worldmodel's train/lewm.py, for
several agents).

    python scripts/train/lewm.py
    python scripts/train/lewm.py inputs=[pixels,position,direction] output_model_name=lewm_findgoal_pos

Reads the dataset <cache_dir>/datasets/<dataset_name> (its metadata gives the env), trains with
LeWM's loss (next-embedding prediction + SIGReg) and saves <cache_dir>/checkpoints/<output_model_name>.pt,
which `DecentralizedWorldModel.load` reads back (see scripts/plan/eval_wm.py).
"""
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, load_dataset
from stable_marl.wm import DecentralizedWorldModel, train_world_model

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    torch.manual_seed(cfg.seed)
    inputs = list(cfg.inputs)
    dataset = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir, num_steps=cfg.wm.history_size + cfg.wm.num_preds,
                           keys_to_load=[*inputs, 'action', 'terminated'])
    meta = dataset.metadata
    env = sm.make(meta['env_name'], **meta['env_kwargs'])
    num_agents, num_actions = meta['num_agents'], int(env.action_space[0].n)
    logger.info(f"{cfg.dataset_name}: {len(dataset.lengths)} episodes, {len(dataset)} clips, {num_agents} agents, "
                f"{num_actions} actions, inputs {inputs}")

    # extra inputs next to the image: their size per agent, e.g. position 2, direction 1
    extra = {k: int(np.prod(dataset.get_col_data(k).shape[2:])) or 1 for k in inputs if k != 'pixels'}
    model_kwargs = OmegaConf.to_container(cfg.model)
    model = DecentralizedWorldModel.build_lewm(num_agents=num_agents, num_actions=num_actions,
                                               history_size=cfg.wm.history_size,
                                               **({'extra_inputs': extra} if extra else {}), **model_kwargs)
    history = train_world_model(model, dataset, epochs=cfg.train.epochs, batch_size=cfg.train.batch_size,
                                lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                                sigreg_weight=cfg.train.sigreg_weight,
                                sigreg_kwargs={'num_proj': cfg.train.sigreg_num_proj},
                                history_size=cfg.wm.history_size, num_preds=cfg.wm.num_preds,
                                val_split=cfg.train.val_split, seed=cfg.seed, device=cfg.device,
                                log=lambda row: logger.info(row))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / f'{cfg.output_model_name}.pt'
    model.save(path)
    logger.info(f"train loss {history['train_loss'][0]:.3f} -> {history['train_loss'][-1]:.3f}, "
                f"val {history['val_loss'][0]:.3f} -> {history['val_loss'][-1]:.3f}; saved {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
