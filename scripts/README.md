# Scripts

Examples of how to use stable-marl, laid out like stable-worldmodel's `scripts/`. Run them from the
repository root (`python scripts/...`).

## Examples (`examples/`)

Short scripts showing one part of the library each, runnable on a CPU in a few minutes. The experiment
(`end_to_end_lewm.py`) is configured with Hydra like the pipeline; the others are plain scripts.

| Script | Shows |
|---|---|
| `quickstart.py` | `World`: pools of multi-agent envs, the info dict, evaluating policies (random, expert), videos of episodes |
| `data_pipeline.py` | collecting datasets (HDF5, folder), reading training clips, `GoalDataset`, `ConcatDataset`, `ReplayBuffer`, `convert` / `merge`, column normalization |
| `variations_and_wrappers.py` | factors of variation (colors, layouts) as reset options, visual wrappers (noise schedules, occlusions, grayscale), panel videos |
| `planning_with_known_dynamics.py` | the planning stack (dynamics, objective, solver, `WorldModelPolicy`) with the exact navigation model, joint vs per-agent planning, solver callbacks |
| `end_to_end_lewm.py` | collecting data, training one LeWM per agent and planning with them, in one script (Hydra: `examples/config/end_to_end_lewm.yaml`, e.g. `env.obs_mode=global inputs=[pixels,position,direction] env.agents=1`) |
| `rware_warehouse.py` | the RWARE environments (multi-robot warehouse): per-robot `vector` and `pixels` observations, the global state, datasets with both, vector-only worlds, videos of the whole warehouse |
| `vmas_scenarios.py` | the VMAS environments (needs `pip install 'stable-marl[vmas]'`): continuous and discrete actions, agents of different kinds, datasets with the scenarios' infos, random policies on a few scenarios |
| `robofactory_tasks.py` | the RoboFactory environments (needs `pip install 'stable-marl[robofactory]'`): 2 to 4 Panda arms, per-arm cameras and joint states, the simulation state and global camera, datasets and dataset-driven evaluation, and (with numpy < 2) the motion-planning expert with videos |
| `communication.py` | agents that communicate, modelled with comm-core (needs `pip install 'stable-marl[comm]'`): a method exchanging world-model latents over perfect and lossy channels (`AgentNetwork`), communication as part of a `World`'s envs (`communication(...)`: received messages in the observations and datasets), a topology following the agents' positions (`within_range`) |
| `generalization.py` | train once on FindGoal (15 x 15, 7 x 7 local views, 6 inner walls), then evaluate under environment changes never seen in training: other colors, farther goals, more inner walls, noisy or occluded views (Hydra: `examples/config/generalization.yaml`; `models=[lewm,gcbc]` by default, any of the `gcrl` baselines too). The eval scripts take the same shifts: `eval.env_kwargs`, `eval.options`, `eval.wrappers` |

## Pipeline (`data/` → `train/` → `plan/`)

Hydra scripts, as in stable-worldmodel; any config key can be overridden on the command line
(`key=value`). Datasets go to `<cache_dir>/datasets/` and checkpoints to `<cache_dir>/checkpoints/`
(`cache_dir`: `$STABLEMARL_HOME`, else `~/.stable_marl`).

    python scripts/data/collect_findgoal.py                 # expert + random FindGoal episodes -> findgoal.h5
    python scripts/data/collect_findgoal.py --config-name mixed   # MixedPolicy episodes -> findgoal_mixed.h5
    python scripts/data/collect_robofactory.py env.task=LiftBarrier   # RoboFactory motion-planning expert (numpy<2) -> robofactory_liftbarrier.h5
    python scripts/train/lewm.py                            # one LeWM per agent -> checkpoints/lewm_findgoal/
    python scripts/plan/eval_wm.py                          # plan with it: success rate, results file, videos
    python scripts/plan/eval_wm.py policy=random            # baselines: random, expert
    python scripts/train/gcrl.py agent=hiql                 # MangoBench's goal-conditioned baselines (JAX) -> checkpoints/hiql_findgoal/
    python scripts/plan/eval_ff.py policy=hiql_findgoal     # evaluate them (feed-forward policies, no planning)
    python scripts/train/mamba.py goal_conditioned=true     # MAMBA, online (shaped reward) or goal-conditioned -> checkpoints/gc_mamba_findgoal/
    python scripts/plan/eval_ff.py policy=gc_mamba_findgoal
    python scripts/train/omnivla.py                         # OmniVLA-edge (torch only) on findgoal_mixed -> checkpoints/omnivla_findgoal_mixed/
    python scripts/plan/eval_ff.py policy=omnivla_findgoal_mixed dataset_name=findgoal_mixed
    python scripts/train/commnet.py                         # CommNet, online (REINFORCE) or mode=supervised -> checkpoints/commnet_online/
    python scripts/plan/eval_ff.py policy=commnet_online
    python scripts/train/mappo.py                           # MAPPO, online (critic=concat|state|local) -> checkpoints/mappo_concat/
    python scripts/plan/eval_ff.py policy=mappo_concat
    python scripts/train/tarmac.py                          # TarMAC, online (targeted, multi-round communication) -> checkpoints/tarmac/
    python scripts/plan/eval_ff.py policy=tarmac
    python scripts/train/commnet.py --config-name commnet_gc  # goal-conditioned CommNet (hindsight goals; shares latents, observations, poses)
    python scripts/plan/eval_ff.py policy=commnet_gc_supervised goal_modalities=[image]

| Script | Config | Notes |
|---|---|---|
| `data/collect_findgoal.py` | `data/config/default.yaml`, `data/config/mixed.yaml` | `env.*` (agents, size, `obs_mode`, ...), `format` (`hdf5`, `folder`, `video`), `policies` (any policy classes with their `episodes`; default: expert and random, `--config-name mixed`: `MixedPolicy`), reset `options` (e.g. variations) |
| `data/collect_robofactory.py` | `data/config/robofactory.yaml` | `env.*` (task, scene, observations, `camera_size`), `expert_episodes`, `only_success`, `random_episodes`, `shard` (one job of an array: own seeds and file, merged with `stable_marl.data.merge`); the expert needs numpy < 2, assets: `download_assets()` beforehand on machines without internet |
| `train/lewm.py` | `train/config/lewm.yaml` | `inputs` (e.g. `[pixels,position,direction]`), model size, `wm.history_size`, `train.*`, `device` |
| `train/gcrl.py` | `train/config/gcrl.yaml` | `agent` (`gcbc`, `crl`, `hiql`, `gcivl`, `gciql`: MangoBench's decentralized baselines, one agent per agent), `obs_keys`, `train.steps`, `agent_config.*` (needs `pip install 'stable-marl[gcrl]'`) |
| `train/omnivla.py` | `train/config/omnivla.yaml` | OmniVLA-edge with a discrete action head, one model shared by the agents: `data.labels` (`shortest_path`: the expert's actions recomputed from the recorded grids, whatever policy collected the data; `collected`), `data.hindsight_prob`, `data.modalities` (`pose`, `image`), `model.*`, `pretrained` (the released weights), `train.*` |
| `train/commnet.py` | `train/config/commnet.yaml` | CommNet (communication by averaging hidden states, one network for all agents): `mode` (`online`: REINFORCE with a baseline in `env.*`, the paper; `supervised`: the expert's actions on `dataset_name`), `gc.*` (`mode=gc_supervised`, `--config-name commnet_gc`: goals per agent, hindsight, `gc.share` of `latent` (quantized: `gc.latent_bits`), `observation`, `pose`), `model.comm_steps` (1: no communication), `model.comm_range`, `model.drop_prob` (lost messages), `comm` (messages over comm-core networks, `NetworkLinks`: `--config-name commnet_erasure`), `obs_keys`, `goal`, `online.*`, `supervised.*` |
| `train/mappo.py` | `train/config/mappo.yaml` | MAPPO (torch only): decentralized shared actor, centralized critic (`critic`: `concat` all observations, `state` the env's state, `local` = IPPO; `agent_specific`), value normalization, death masking, `train.*` (PPO), `model.comm_range` / `model.drop_prob` / `comm` (the critics' observations over comm-core) |
| `train/tarmac.py` | `train/config/tarmac.yaml` | TarMAC (torch only): recurrent agents, messages with a signature and a value attended to by queries, `model.rounds`, `model.latent_bits` (quantized messages), `model.comm_range` / `model.drop_prob` / `comm` (comm-core); the paper's actor-critic with a centralized critic (`train.*`) |
| `train/mamba.py` | `train/config/mamba.yaml` | `goal_conditioned` (false: MAMBA with its Flatland-style shaped reward; true: GC-MAMBA with goal offsets and hindsight relabelling), `env.*` (trained online, no dataset), `train.episodes`, `rewards.*`, `mamba_config.*` (MAMBA's settings) |
| `plan/eval_ff.py` | `plan/config/eval_ff.yaml` | `policy` (a checkpoint of `train/gcrl.py`, `train/mamba.py`, `train/omnivla.py`, `train/commnet.py`, `train/mappo.py` or `train/tarmac.py`), `temperature`, `goal_modalities` (OmniVLA, GCCommNet), `comm` (CommNet, TarMAC: another comm-core channel when acting), `eval.*` (incl. shifts from the training settings: `eval.env_kwargs`, `eval.options`, `eval.wrappers`) |
| `plan/eval_wm.py` | `plan/config/findgoal.yaml`, `solver/`, `objective/` | `policy` (a checkpoint, `random` or `expert`), `mode` (`per_agent` or `joint`), `solver=categorical_cem` / `categorical_mppi`, `plan_config.*`, `eval.*` |

Generated videos (`examples/videos/`) and results (`plan/outputs/`) are not tracked by git.

## Other scripts

`log_obs_modes.py` logs the `ego` / `allo` / `global` observation modes side by side (Hydra:
`config/log_obs_modes.yaml`, e.g. `python scripts/log_obs_modes.py env=MultiGrid-FindGoal-15x15-v0 steps=40`).
`a_star.py`, `coordinate_mapping.py`, `export_grid.py`, `blender_scene.py` (environment tools: shortest
paths, coordinates, exports), `test*.py` and `find_goal_test.py` (manual checks of the envs and their
external wrappers), and the notebooks.

## Training MultiGrid agents with RLlib

MultiGrid is compatible with RLlib's multi-agent API.

This folder provides scripts to train and visualize agents over MultiGrid environments.

### Requirements

Using MultiGrid environments with RLlib requires installation of [rllib](https://docs.ray.io/en/latest/rllib/index.html), and one of [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

### Getting Started

Train 2 agents on the `MultiGrid-Empty-8x8-v0` environment using the PPO algorithm (Hydra config:
`config/train.yaml`; e.g. `num_workers=0 num_gpus=0` on a CPU, `+env_config.size=8` for env kwargs):

    python scripts/train.py algo=PPO env=MultiGrid-Empty-8x8-v0 num_agents=2 save_dir=~/saved/empty8x8/

Visualize behavior from trained agents policies: see the notebook `visualize.ipynb`.

For the options, see `config/train.yaml` (or run ``python scripts/train.py --help``).

### Environments

All of the environment configurations registered in [`stable_marl.envs`](../stable_marl/envs/__init__.py)
can also be used with RLlib, registered under the same names by `register_rllib_envs`:

    >>> from stable_marl.wrappers.external import register_rllib_envs
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> register_rllib_envs()
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

To convert a custom `MultiGridEnv` to an RLlib `MultiAgentEnv`:

    >>> from stable_marl.wrappers.external import to_rllib_env
    >>> MyRLLibEnvClass = to_rllib_env(MyEnvClass)
    >>> algorithm_config = PPOConfig().environment(env=MyRLLibEnvClass)
