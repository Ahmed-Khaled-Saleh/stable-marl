# Vendored MAMBA (PyTorch)

MAMBA, "Scalable Multi-Agent Model-Based Reinforcement Learning" (Egorov & Shpilman, AAMAS 2022):
a DreamerV2-style world model per agent (shared weights) whose transition attends over the agents
(communication at execution), with actor and critic trained by PPO in imagination.
`stable_marl.wm.mamba` runs it on MultiGrid envs.

Source: [jbr-ai-labs/mamba](https://github.com/jbr-ai-labs/mamba), commit `2c97258` (MIT, see LICENSE).

| here | source |
|---|---|
| `agent/models`, `agent/learners`, `agent/memory`, `agent/controllers`, `agent/optim`, `agent/utils` | same paths |
| `networks/dreamer`, `networks/transformer` | same paths |
| `configs/Config.py`, `configs/dreamer/*Config.py` | same paths (not the `optimal/` copies nor the Flatland / SMAC configs) |
| `environments.py` | same path |

Not vendored: the ray runner and workers (`agent/runners`, `agent/workers`), which drive SMAC and Flatland;
`stable_marl.wm.mamba` replaces them with a loop over MultiGrid envs that fills the same buffers
(`DreamerController.update_buffer` / `dispatch_buffer`, `DreamerLearner.step`).

Changes:
- imports made package-relative, `__init__.py` files added;
- `log.py` (new): a stand-in for `wandb`, which `agent/optim/loss.py` and `agent/learners/DreamerLearner.py`
  log to unconditionally; both import it instead of `wandb`, and the learner no longer creates `LOG_FOLDER`.
