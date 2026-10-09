# Vendored RWARE

The multi-robot warehouse environment (Christianos et al.), from
[semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse), commit
`96fbc64e3eae5fee915e0d390f864fa06ddccd47` (version 2.0.0 on PyPI), MIT license (see LICENSE).

| here | source |
|---|---|
| `warehouse.py` | `rware/warehouse.py`, unchanged |

Not vendored: `rware/rendering.py` (pyglet / OpenGL; `stable_marl.envs.rware` draws its images with numpy,
so `Warehouse.render`, which imports it, is never called), `rware/__init__.py` (gymnasium ids: stable-marl
registers its own), `rware/utils`, `human_play.py`.

Dependencies: numpy, gymnasium, networkx.
