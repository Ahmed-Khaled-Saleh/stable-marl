"""
Train MAPPO (Yu et al., 2022; stable_marl.wm.mappo) online on a MultiGrid env: PPO with a decentralized actor
shared by the agents and a centralized critic (training only), value normalization and death masking.

    python scripts/train/mappo.py                                   # centralized critic on all the observations
    python scripts/train/mappo.py critic=state                      # on the env's global state
    python scripts/train/mappo.py critic=local                      # IPPO
    python scripts/train/mappo.py model.drop_prob=0.3 model.comm_range=3      # lossy observations for the critics
    python scripts/train/mappo.py 'comm={_target_: stable_marl.wm.commnet.NetworkLinks, channel: {_target_: comm_core.ErasureChannel, loss_rate: 0.2}}'

Saves <cache_dir>/checkpoints/<output_model_name>/mappo.pt (with the env it was trained on), which
scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir
from stable_marl.wm.commnet import commnet_features
from stable_marl.wm.mappo import MAPPO, train_mappo

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='mappo')
def run(cfg):
    torch.manual_seed(cfg.seed)
    env_kwargs = OmegaConf.to_container(cfg.env)
    env_name = env_kwargs.pop('name')
    world = sm.World(env_name, num_envs=cfg.num_envs, goal_conditioned=True, **env_kwargs)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: v[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    obs_dim = commnet_features(columns, cfg.obs_keys, env.width, cfg.goal).shape[-1]
    model = MAPPO(obs_dim, world.num_agents, num_actions=int(env.action_space[0].n), critic=cfg.critic,
                  agent_specific=cfg.agent_specific, state_dim=int(np.prod(columns['state'].shape[1:])),
                  obs_keys=list(cfg.obs_keys), goal=cfg.goal, grid_size=env.width, **OmegaConf.to_container(cfg.model))
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if cfg.get('comm') is not None:                              # the critics' observations over comm-core
        model.links = hydra.utils.instantiate(cfg.comm, num_agents=world.num_agents)
    logger.info(f"MAPPO (critic={cfg.critic}{', agent-specific' if model.agent_specific else ''}) on {env_name} "
                f"{env_kwargs}: {world.num_agents} agents, {obs_dim} features, "
                f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters, on {device}")
    train_mappo(model, world, seed=cfg.seed, log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}),
                **OmegaConf.to_container(cfg.train))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'mappo.pt'
    model.cpu().save(path, env_name=env_name, env_kwargs=env_kwargs)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
