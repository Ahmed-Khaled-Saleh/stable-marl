"""
Datasets: collect episodes to disk, read them back as training clips, sample goals, keep them in
memory, convert and merge them, normalize columns.

    python scripts/examples/data_pipeline.py

Every info key becomes a column with one row per step (stable-worldmodel's layout). Readers return
clips of `num_steps` steps as torch tensors; per-agent columns keep their agent axis.
"""
import tempfile
from pathlib import Path

import torch

import stable_marl as sm
from stable_marl.data import (ConcatDataset, GoalDataset, ReplayBuffer, column_normalizer, convert, list_formats,
                              load_dataset, merge)
from stable_marl.envs.multigrid import GoToGoalPolicy

ENV = dict(agents=2, size=7, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1, tile_size=8)
world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, max_episode_steps=20, goal_conditioned=True, **ENV)
print('formats:', list_formats())

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # 1. collect: expert and random episodes, in two formats
    world.set_policy(GoToGoalPolicy())
    world.collect(tmp / 'expert.h5', episodes=12, seed=0, progress=False)            # one HDF5 file
    world.set_policy(sm.RandomPolicy(seed=0))
    world.collect(tmp / 'random', episodes=6, seed=1_000, format='folder', progress=False)   # npz + images

    # 2. read: clips of 4 steps (`load_dataset` detects the format; kwargs go to the reader)
    expert = load_dataset(tmp / 'expert.h5', num_steps=4, keys_to_load=['pixels', 'position', 'action', 'terminated'])
    print(f'expert: {len(expert.lengths)} episodes, {len(expert)} clips, metadata {expert.metadata}')
    clip = expert[0]
    print({k: tuple(v.shape) for k, v in clip.items()})     # pixels (4, agents, H, W, 3), action (4, agents)

    # 3. a goal per clip, sampled from a later step of the same episode (or a random / the current one)
    goals = GoalDataset(expert, goal_probabilities=(0.2, 0.6, 0.0, 0.2), seed=0)
    item = goals[0]
    print('goal columns:', {k: tuple(v.shape) for k, v in item.items() if k.startswith('goal_')})

    # 4. several datasets as one, for a DataLoader
    random = load_dataset(tmp / 'random', num_steps=4, keys_to_load=['pixels', 'position', 'action', 'terminated'])
    both = ConcatDataset([expert, random])
    batch = next(iter(torch.utils.data.DataLoader(both, batch_size=8, shuffle=True)))
    print('batch:', {k: tuple(v.shape) for k, v in batch.items()})

    # 5. in memory: a replay buffer is a writer (fill it with collect) and a dataset (train on it)
    buffer = ReplayBuffer(max_steps=500, history_len=4)
    world.set_policy(GoToGoalPolicy())
    world.collect(writer=buffer, episodes=8, seed=0, progress=False)
    print(buffer, '| sample:', {k: v.shape for k, v in buffer.sample(16).items() if k in ('pixels', 'action')})
    buffer.dump(tmp / 'buffer.h5', format='hdf5')

    # 6. convert and merge files, normalize a column
    convert(tmp / 'random', tmp / 'random.h5', dest_format='hdf5', progress=False)
    merge([tmp / 'expert.h5', tmp / 'random.h5'], tmp / 'all.h5', progress=False)
    everything = load_dataset(tmp / 'all.h5', num_steps=4)
    print(f'merged: {len(everything.lengths)} episodes')
    norm = column_normalizer(everything, 'position', 'position_norm')     # z-score, fitted on the column
    print('normalized position of a clip:', norm(everything[0])['position_norm'][0].tolist())
    for ds in (expert, random, everything):
        ds.close() if hasattr(ds, 'close') else None
