"""
Train DreamerV3 (stable_marl.wm.dreamer: r2dreamer's PyTorch implementation, vendored) online on a MultiGrid
env, every agent a row of one world model and actor-critic (shared weights).

    python scripts/train/dreamer.py
    python scripts/train/dreamer.py model.size=25M 'model.overrides={deter: null, hidden: null, units: null}' device=cuda train.train_ratio=512
    python scripts/train/dreamer.py model.rep_loss=r2dreamer        # R2-Dreamer's decoder-free world model
    python scripts/train/dreamer.py image_key=pixels obs_keys=[direction]   # each agent's RGB view, through the CNNs

Saves <cache_dir>/checkpoints/<output_model_name>/dreamer.pt (with the env it was trained on), which
scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir
from stable_marl.wm.commnet import commnet_features
from stable_marl.wm.dreamer import DreamerAgents, dreamer_config, train_dreamer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='dreamer')
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
    model = DreamerAgents(config, obs_dim, world.num_agents, num_actions=int(env.action_space[0].n),
                          obs_keys=list(cfg.obs_keys), goal=cfg.goal, grid_size=env.width, team_reward=cfg.team_reward,
                          image_key=cfg.image_key, image_size=cfg.image_size)
    logger.info(f"DreamerV3 ({cfg.model.rep_loss}, size {cfg.model.size} with {overrides}) on {env_name} "
                f"{env_kwargs}: {world.num_agents} agents, {obs_dim} features, "
                f"{sum(p.numel() for p in model.agent._named_params.values()) / 1e6:.2f}M parameters, on {device}")
    train_dreamer(model, world, seed=cfg.seed, log=lambda row: logger.info({k: round(v, 4) for k, v in row.items()}),
                  **OmegaConf.to_container(cfg.train))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'dreamer.pt'
    model.save(path, env_name=env_name, env_kwargs=env_kwargs)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
