"""
Train TarMAC (Das et al., 2019; stable_marl.wm.tarmac) online on a MultiGrid env: recurrent agents sending
messages with a signature and a value, attended to by the receivers' queries, over one or more rounds per
step; the paper's multi-agent actor-critic with a centralized critic, RMSProp, batches of 16 episodes.

    python scripts/train/tarmac.py
    python scripts/train/tarmac.py model.rounds=2                   # 2-round communication
    python scripts/train/tarmac.py model.latent_bits=8 model.comm_range=3
    python scripts/train/tarmac.py 'comm={_target_: stable_marl.wm.commnet.NetworkLinks, channel: {_target_: comm_core.ErasureChannel, loss_rate: 0.2}}'

Saves <cache_dir>/checkpoints/<output_model_name>/tarmac.pt (with the env it was trained on), which
scripts/plan/eval_ff.py evaluates (over the same channel, or another: eval_ff.py comm=...).
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir
from stable_marl.wm.commnet import commnet_features
from stable_marl.wm.tarmac import TarMAC, train_tarmac

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='tarmac')
def run(cfg):
    torch.manual_seed(cfg.seed)
    env_kwargs = OmegaConf.to_container(cfg.env)
    env_name = env_kwargs.pop('name')
    world = sm.World(env_name, num_envs=cfg.num_envs, goal_conditioned=True, **env_kwargs)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: v[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    obs_dim = commnet_features(columns, cfg.obs_keys, env.width, cfg.goal).shape[-1]
    model = TarMAC(obs_dim, world.num_agents, num_actions=int(env.action_space[0].n), obs_keys=list(cfg.obs_keys),
                   goal=cfg.goal, grid_size=env.width, **OmegaConf.to_container(cfg.model))
    model.to(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if cfg.comm is not None:                                     # messages over comm-core networks
        model.links = hydra.utils.instantiate(cfg.comm, num_agents=world.num_agents)
    logger.info(f"TarMAC ({model.rounds} round(s), {model.message_bits} bits per message) on {env_name} {env_kwargs}: "
                f"{world.num_agents} agents, {obs_dim} features, {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    train_tarmac(model, world, seed=cfg.seed, log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}),
                 **OmegaConf.to_container(cfg.train))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'tarmac.pt'
    comm = OmegaConf.to_container(cfg.comm, resolve=True) if cfg.comm is not None else None
    model.cpu().save(path, env_name=env_name, env_kwargs=env_kwargs, comm=comm)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
