# Vendored goal-conditioned RL agents (JAX)

The goal-conditioned offline RL agents that MangoBench (CVPR 2026) trains once per agent for its
fully decentralized baselines (GCMBC = GCBC, ICRL = CRL, IHIQL = HIQL, IGCIVL = GCIVL), plus GCIQL.
They are OGBench's single-agent implementations; `stable_marl.wm.gcrl` adds the multi-agent part
(one independent agent per MultiGrid agent, as MangoBench's `MultiAgentManager`).

| file | source | commit |
|---|---|---|
| `agents/gcbc.py`, `agents/crl.py`, `agents/hiql.py` | [mangobench-locomotion](https://github.com/SYSU-SAIL/mangobench-locomotion) `impls/agents/` | `236a83cdd66df5376249e2514b4ef9c6b47936e2` |
| `utils/networks.py`, `utils/encoders.py`, `utils/flax_utils.py`, `utils/datasets.py` | mangobench-locomotion `impls/utils/` | same |
| `agents/gcivl.py` | [mangobench-manipulation](https://github.com/WendyeeWang/mangobench-manipulation) `robofactory/policy/OGCRL/ogcrl/agents/` | `1544c92cca0d167b3fd38c8ebe9221384efbdb8e` |
| `agents/gciql.py` | [OGBench](https://github.com/seohongpark/ogbench) `impls/agents/` (MangoBench's copy has `@jax.jit` commented out) | `a977253ce5b88434f4e4f531c50047cc95993a51` |

Changes: only the imports (`from utils.x` / `from ogcrl.utils.x` -> package-relative imports), and
the `__init__.py` files. Do not edit these files; adapt them in `stable_marl.wm.gcrl` instead.

Notes on the sources: MangoBench's GCBC is OGBench's; its HIQL differs from OGBench's only by comments;
its CRL is an earlier OGBench version (the actor maximizes log Q, `actor_log_q=True`).
