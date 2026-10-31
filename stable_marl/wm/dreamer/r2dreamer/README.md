# Vendored r2dreamer (PyTorch)

The model code of [NM512/r2dreamer](https://github.com/NM512/r2dreamer) (commit `becf338`, MIT, see LICENSE): the
official implementation of R2-Dreamer (ICLR 2026), which includes an efficient PyTorch reproduction of DreamerV3
(`rep_loss: dreamer`), and the R2-Dreamer, InfoNCE and DreamerPro representation losses (`rep_loss: r2dreamer`,
`infonce`, `dreamerpro`). `stable_marl.wm.dreamer` runs it on MultiGrid envs, one agent per batch row.

| here | source |
|---|---|
| `dreamer.py`, `rssm.py`, `networks.py`, `distributions.py` | same paths |
| `optim/` (LaProp, adaptive gradient clipping) | same path |
| `tools.py` | `tools.py`: only `to_np`, `to_f32`, `to_i32`, `weight_init_`, `tensorstats`, `compute_rms`, `compute_global_norm`, `rpad` |
| `configs/model/*.yaml` | same paths |

Not vendored: the training loop and envs (`train.py`, `trainer.py`, `envs/`), the replay buffer (`buffer.py`,
on TorchRL), logging (TensorBoard). `stable_marl.wm.dreamer` provides the loop, and a replay with the same
`sample()` / `update()` interface as `buffer.py`'s `Buffer`.

Changes, each marked `# stable-marl:` in the code:

* package-relative imports; `logging` instead of `print`;
* no `tensordict`: `Dreamer.act` / `get_initial_state` return plain dicts, and `_cal_grad` reads the batch shape
  from `data["action"]`;
* `Dreamer._cal_grad` takes an optional `mask` (B, T, 1) in the batch: the world model's losses (dynamics,
  representation, reconstruction or R2-Dreamer / InfoNCE, reward, continuation), the imagination starts and the
  replay value loss only count the steps where it is 1 (with unbiased means). Without `mask`, it is unchanged.
  stable-marl uses it for agents whose episode ended (e.g. they reached their goal) while the others act on.
