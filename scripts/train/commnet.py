"""
Train CommNet (Sukhbaatar et al., 2016; stable_marl.wm.commnet): one network for all the agents, whose
hidden states are averaged across agents (communication) at every step. Online with REINFORCE and a
learned baseline (the paper), or from a dataset with the shortest-path expert's actions as labels; and its
goal-conditioned version (GCCommNet: a goal per agent, hindsight goals, sharing quantized latents, raw
observations and/or poses), from a dataset.

    python scripts/train/commnet.py                                         # online, fully connected
    python scripts/train/commnet.py model.comm_range=3 model.drop_prob=0.2  # local, lossy communication
    python scripts/train/commnet.py model.comm_steps=1                      # no communication (the paper's baseline)
    python scripts/train/commnet.py --config-name commnet_erasure comm.channel.loss_rate=0.5   # over comm-core
    python scripts/data/collect_findgoal.py --config-name mixed
    python scripts/train/commnet.py mode=supervised                         # from findgoal_mixed
    python scripts/train/commnet.py --config-name commnet_gc                # goal-conditioned, sharing latents, observations, poses
    python scripts/train/commnet.py --config-name commnet_gc 'gc.share=[pose]' gc.hindsight=agent

Saves <cache_dir>/checkpoints/<output_model_name>/commnet.pt (with the env it was trained on), which
scripts/plan/eval_ff.py evaluates.
"""
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

import stable_marl as sm
from stable_marl.data import get_cache_dir, load_dataset
from stable_marl.wm.commnet import (CommNet, CommNetDataset, GCCommNet, GCCommNetDataset, commnet_features, train_commnet,
                                    train_commnet_supervised, train_gc_commnet)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='./config', config_name='commnet')
def run(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.mode in ('supervised', 'gc_supervised'):
        dataset = load_dataset(cfg.dataset_name, cache_dir=cfg.cache_dir)
        env_name, env_kwargs = dataset.metadata['env_name'], dataset.metadata['env_kwargs']
    elif cfg.mode == 'online':
        env_kwargs = OmegaConf.to_container(cfg.env)
        env_name = env_kwargs.pop('name')
    else:
        raise ValueError(f"mode must be online, supervised or gc_supervised, got {cfg.mode!r}")
    world = sm.World(env_name, num_envs=cfg.num_envs, goal_conditioned=True, **env_kwargs)
    world.reset(seed=cfg.seed)
    env = world.envs.envs[0].unwrapped
    columns = {k: v[:, -1] for k, v in world.infos.items() if not k.startswith('_')}
    num_actions, model_kwargs = int(env.action_space[0].n), OmegaConf.to_container(cfg.model)
    if cfg.mode == 'gc_supervised':
        gc = OmegaConf.to_container(cfg.gc)
        obs_dim = commnet_features(columns, cfg.obs_keys, env.width, goal=False).shape[-1]
        model = GCCommNet(obs_dim, goal_image_cells=int(env.agents[0].view_size), num_actions=num_actions, obs_keys=list(cfg.obs_keys),
                          grid_size=env.width, share=gc['share'], latent_bits=gc['latent_bits'],
                          bits_per_element=gc['bits_per_element'], goal_modalities=gc['goal_modalities'], **model_kwargs)
    else:
        obs_dim = commnet_features(columns, cfg.obs_keys, env.width, cfg.goal).shape[-1]
        model = CommNet(obs_dim, num_actions=num_actions, obs_keys=list(cfg.obs_keys), goal=cfg.goal, grid_size=env.width,
                        **model_kwargs)
    logger.info(f"CommNet ({cfg.mode}) on {env_name} {env_kwargs}: {world.num_agents} agents, {obs_dim} features, "
                f"{sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    if cfg.comm is not None:                                     # messages over comm-core networks
        model.links = hydra.utils.instantiate(cfg.comm, num_agents=world.num_agents)
        logger.info(f'messages over comm-core: {OmegaConf.to_container(cfg.comm)}')
    log = lambda row: logger.info({k: round(v, 4) for k, v in row.items()})

    if cfg.mode == 'online':
        train_commnet(model, world, seed=cfg.seed, log=log, **OmegaConf.to_container(cfg.online))
    elif cfg.mode == 'supervised':
        data = CommNetDataset(dataset, model)
        logger.info(f'{cfg.dataset_name}: {len(dataset.lengths)} episodes, {len(data)} joint samples')
        train_commnet_supervised(model, data, log=log, **OmegaConf.to_container(cfg.supervised))
    else:
        data = GCCommNetDataset(dataset, model, hindsight_prob=cfg.gc.hindsight_prob, hindsight=cfg.gc.hindsight,
                                max_goal_steps=cfg.gc.max_goal_steps)
        logger.info(f'{cfg.dataset_name}: {len(dataset.lengths)} episodes, {len(data)} joint samples; agents share '
                    f'{list(model.share) or "nothing"}, {model.message_bits(columns)} bits per round')
        train_gc_commnet(model, data, log=log, **OmegaConf.to_container(cfg.supervised))
    path = Path(get_cache_dir(cfg.cache_dir, sub_folder='checkpoints')) / cfg.output_model_name / 'commnet.pt'
    comm = OmegaConf.to_container(cfg.comm, resolve=True) if cfg.comm is not None else None
    model.cpu().save(path, env_name=env_name, env_kwargs=env_kwargs, comm=comm)
    if model.links is not None and cfg.mode == 'online':
        logger.info(f'comm-core metrics of the last episodes: {model.links.metrics}')
    logger.info(f'saved {path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
    run()
