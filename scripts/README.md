# Scripts

Examples of how to use stable-marl, laid out like stable-worldmodel's `scripts/`. Run them from the
repository root (`python scripts/...`).

## Examples (`examples/`)

Short scripts showing one part of the library each, runnable on a CPU in a few minutes.

| Script | Shows |
|---|---|
| `quickstart.py` | `World`: pools of multi-agent envs, the info dict, evaluating policies (random, expert), videos of episodes |
| `data_pipeline.py` | collecting datasets (HDF5, folder), reading training clips, `GoalDataset`, `ConcatDataset`, `ReplayBuffer`, `convert` / `merge`, column normalization |
| `variations_and_wrappers.py` | factors of variation (colors, layouts) as reset options, visual wrappers (noise schedules, occlusions, grayscale), panel videos |
| `planning_with_known_dynamics.py` | the planning stack (dynamics, objective, solver, `WorldModelPolicy`) with the exact navigation model, joint vs per-agent planning, solver callbacks |
| `end_to_end_lewm.py` | collecting data, training one LeWM per agent and planning with them, in one script (`--obs-mode ego/allo/global`, `--position`, `--direction`, `--agents`) |

## Pipeline (`data/` → `train/` → `plan/`)

Hydra scripts, as in stable-worldmodel; any config key can be overridden on the command line
(`key=value`). Datasets go to `<cache_dir>/datasets/` and checkpoints to `<cache_dir>/checkpoints/`
(`cache_dir`: `$STABLEMARL_HOME`, else `~/.stable_marl`).

    python scripts/data/collect_findgoal.py                 # expert + random FindGoal episodes -> findgoal.h5
    python scripts/train/lewm.py                            # one LeWM per agent -> checkpoints/lewm_findgoal.pt
    python scripts/plan/eval_wm.py                          # plan with it: success rate, results file, videos
    python scripts/plan/eval_wm.py policy=random            # baselines: random, expert

| Script | Config | Notes |
|---|---|---|
| `data/collect_findgoal.py` | `data/config/default.yaml` | `env.*` (agents, size, `obs_mode`, ...), `format` (`hdf5`, `folder`, `video`), episodes per policy, reset `options` (e.g. variations) |
| `train/lewm.py` | `train/config/lewm.yaml` | `inputs` (e.g. `[pixels,position,direction]`), model size, `wm.history_size`, `train.*`, `device` |
| `plan/eval_wm.py` | `plan/config/findgoal.yaml`, `solver/`, `objective/` | `policy` (a checkpoint, `random` or `expert`), `mode` (`per_agent` or `joint`), `solver=categorical_cem` / `categorical_mppi`, `plan_config.*`, `eval.*` |

Generated videos (`examples/videos/`) and results (`plan/outputs/`) are not tracked by git.

## Other scripts

`a_star.py`, `coordinate_mapping.py`, `export_grid.py`, `blender_scene.py`, `log_obs_modes.py`
(environment tools: shortest paths, coordinates, exports, observation modes), `test*.py` and
`find_goal_test.py` (manual checks of the envs and their external wrappers), and the notebooks.

## Training MultiGrid agents with RLlib

MultiGrid is compatible with RLlib's multi-agent API.

This folder provides scripts to train and visualize agents over MultiGrid environments.

### Requirements

Using MultiGrid environments with RLlib requires installation of [rllib](https://docs.ray.io/en/latest/rllib/index.html), and one of [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

### Getting Started

Train 2 agents on the `MultiGrid-Empty-8x8-v0` environment using the PPO algorithm:

    python train.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --save-dir ~/saved/empty8x8/

Visualize behavior from trained agents policies:

    python visualize.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --load-dir ~/saved/empty8x8/

For more options, run ``python train.py --help`` and ``python visualize.py --help``.

### Environments

All of the environment configurations registered in [`stable_marl.envs`](../stable_marl/envs/__init__.py) can also be used with RLlib, and are registered via `import stable_marl.rllib`.

To use a specific MultiGrid environment configuration by name:

    >>> import stable_marl.rllib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

To convert a custom `MultiGridEnv` to an RLlib `MultiAgentEnv`:

    >>> from stable_marl.rllib import to_rllib_env
    >>> MyRLLibEnvClass = to_rllib_env(MyEnvClass)
    >>> algorithm_config = PPOConfig().environment(env=MyRLLibEnvClass)
