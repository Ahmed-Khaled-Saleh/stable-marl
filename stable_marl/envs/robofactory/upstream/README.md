# Vendored RoboFactory

The multi-arm manipulation tasks of RoboFactory (Qin et al., 2025, "RoboFactory: Exploring Embodied Agent
Collaboration with Compositional Constraints"), from [MARS-EAI/RoboFactory](https://github.com/MARS-EAI/RoboFactory),
commit `5868242322414a91454e22f1dd9641f613ba1bcf`, MIT license (see LICENSE). They are also the manipulation
environments of MangoBench ([SYSU-SAIL/mangobench-manipulation](https://github.com/SYSU-SAIL/mangobench-manipulation),
whose `robofactory/` folder holds the same tasks, scenes and planner).

| here | source |
|---|---|
| `tasks/` | `robofactory/tasks/`: the ManiSkill envs (`LiftBarrier-rf`, ...) |
| `planner/` | `robofactory/planner/`: the motion-planning experts (`solutions/`) and RoboFactory's data generation (`run.py`) |
| `utils/` | `robofactory/utils/`: scene builders (table, RoboCasa), mplib planner, recording wrapper |
| `configs/` | `robofactory/configs/`: scenes, objects, arms and cameras of each task |
| `__init__.py` | `robofactory/__init__.py` |

Changes, all marked `stable-marl` where they are not imports:

* imports of `robofactory.*` made relative;
* `__init__.py`: `ASSET_DIR` is `stable_marl.envs.robofactory.asset_dir()` (`$ROBOFACTORY_ASSET_DIR`, else
  `~/.stable_marl/robofactory/assets`) instead of a folder inside the package;
* `tasks/take_photo.py`: two debug `print`s run at every step commented out;
* empty `__init__.py` added to `utils/`, `utils/wrappers/`, `utils/scenes/robocasa/utils/` and
  `utils/scenes/robocasa/fixtures/` (namespace folders upstream), so that they are installed.

Not vendored: `robofactory/policy/` (Diffusion Policy), `robofactory/script/` (asset download, data
conversion), `utils/scenes/table/assets/` (copies of the table meshes, also in the downloaded assets) and the
3D assets themselves (`sparklexfantasy/RoboFactory_asset` on Hugging Face, downloaded by
`stable_marl.envs.robofactory.download_assets`).

Dependencies: ManiSkill >= 3.0.1 (written for 3.0.0b12; the only incompatibility found, gymnasium 1.x no
longer forwarding attributes through wrappers, is handled by `stable_marl.envs.robofactory`, which gives the
planner the unwrapped env). The planner uses mplib 0.1.1 (pinned by ManiSkill), which needs numpy < 2.
