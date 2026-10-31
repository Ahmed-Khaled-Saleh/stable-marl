"""
Train CoDreamer (Toledo & Prorok, 2024; stable_marl.wm.codreamer) online on a MultiGrid env: DreamerV3
(stable_marl.wm.dreamer) whose agents communicate with GATv2 graph attention inside the world model (RSSM,
reward and continuation heads) and in the actor-critic, over the agents within range.

    python scripts/train/codreamer.py
    python scripts/train/codreamer.py comm_range=3 drop_prob=0.2
    python scripts/train/codreamer.py image_key=pixels obs_keys=[direction]   # each agent's RGB view, through the CNNs
    python scripts/train/codreamer.py 'comm={_target_: stable_marl.wm.commnet.NetworkLinks, channel: {_target_: comm_core.ErasureChannel, loss_rate: 0.2}}'

Saves <cache_dir>/checkpoints/<output_model_name>/codreamer.pt (with the env it was trained on and its channel),
which scripts/plan/eval_ff.py evaluates (over the same channel, or another: eval_ff.py comm=...).
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir
from stable_marl.wm.codreamer import CoDreamerAgents
from stable_marl.wm.commnet import commnet_features
from stable_marl.wm.dreamer import dreamer_config, train_dreamer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='codreamer')
def run(cfg):
    torch.manual_seed(cfg.seed)
    env_kwargs = OmegaConf.to_container(cfg.env)
    env_name = env_kwargs.pop('name')
    world = sm.World(env_name, num_envs=cfg.num_envs, goal_conditioned=True, **env_kwargs)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: v[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    vector = list(cfg.obs_keys) or cfg.goal                          # vector observations, besides an image
    obs_dim = commnet_features(columns, cfg.obs_keys, env.width, cfg.goal).shape[-1] if vector else 0
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    overrides = {k: v for k, v in OmegaConf.to_container(cfg.model.overrides).items() if v is not None}
    config = dreamer_config(cfg.model.size, cfg.model.rep_loss, device, **overrides)
    model = CoDreamerAgents(config, obs_dim, world.num_agents, num_actions=int(env.action_space[0].n),
                            obs_keys=list(cfg.obs_keys), goal=cfg.goal, grid_size=env.width, team_reward=cfg.team_reward,
                            image_key=cfg.image_key, image_size=cfg.image_size,
                            comm_range=cfg.comm_range, drop_prob=cfg.drop_prob, gnn_dim=cfg.gnn_dim)
    if cfg.comm is not None:                                     # messages over comm-core networks
        model.links = hydra.utils.instantiate(cfg.comm, num_agents=world.num_agents)
    logger.info(f"CoDreamer ({cfg.model.rep_loss}, size {cfg.model.size} with {overrides}; range {cfg.comm_range}, "
                f"{model.message_bits} bits per round) on {env_name} {env_kwargs}: {world.num_agents} agents, {obs_dim} "
                f"features, {sum(p.numel() for p in model.agent._named_params.values()) / 1e6:.2f}M parameters, on {device}")
    train_dreamer(model, world, seed=cfg.seed, log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}),
                  **OmegaConf.to_container(cfg.train))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'codreamer.pt'
    comm = OmegaConf.to_container(cfg.comm, resolve=True) if cfg.comm is not None else None
    model.save(path, env_name=env_name, env_kwargs=env_kwargs, comm=comm)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
