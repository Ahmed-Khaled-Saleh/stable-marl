"""
Test the MultiGrid Hydra configs (stable_marl/configs, nbs/04_configs.env.ipynb).

Checks, for every env config:
  1. each default matches the env's constructor default (or MultiGridEnv's, for pass-through args)
  2. the generated YAML file is identical to the dataclass
  3. instantiating the config gives the same env as constructing it with no arguments
  4. composing with Hydra works both ways: YAML via `pkg://stable_marl.configs`
     and dataclasses via `register_configs()`, including overrides and wrappers

Run:  python tests/test_hydra_configs.py
"""
import contextlib
import dataclasses
import inspect
import io
import os
import tempfile

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from stable_marl.configs.env import CONFIG_DIR, ENV_CONFIGS, register_configs
from stable_marl.envs.multigrid.base import MultiGridEnv

HYDRA_KEYS = {'_target_', '_convert_'}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def signature_defaults(cls):
    params = inspect.signature(cls.__init__).parameters
    return {k: p.default for k, p in params.items() if p.default is not inspect.Parameter.empty}


def env_summary(env, seed=0):
    """Things that should match between two identically configured envs."""
    quiet(env.reset, seed=seed)
    return dict(
        width=env.width, height=env.height, max_steps=env.max_steps, num_agents=env.num_agents,
        joint_reward=env.joint_reward, success=env.success_termination_mode,
        failure=env.failure_termination_mode, overlap=env.allow_agent_overlap,
        view=env.agents[0].view_size, see_through=env.agents[0].see_through_walls,
        obs_mode=env.obs_mode, tile_size=env.tile_size, mission=str(env.mission), layout=str(env),
    )


def test_defaults_and_yaml():
    base_defaults = signature_defaults(MultiGridEnv)
    for name, config_cls in ENV_CONFIGS.items():
        env_cls = get_class(config_cls._target_)
        env_defaults = signature_defaults(env_cls)
        node = OmegaConf.structured(config_cls)

        # 1. defaults match the constructors
        for f in dataclasses.fields(config_cls):
            if f.name in HYDRA_KEYS:
                continue
            assert f.name in env_defaults or f.name in base_defaults, f"{name}: unknown arg {f.name}"
            expected = env_defaults.get(f.name, base_defaults.get(f.name))
            value = node[f.name]
            value = list(value) if OmegaConf.is_list(value) else value
            expected = list(expected) if isinstance(expected, tuple) else expected
            assert value == expected, f"{name}.{f.name}: config={value!r} env default={expected!r}"

        # 2. YAML == dataclass
        yaml_node = OmegaConf.load(os.path.join(CONFIG_DIR, 'env', f'{name}.yaml'))
        assert yaml_node == node, f"{name}.yaml differs from {config_cls.__name__}"

        # 3. instantiate(config) == env constructed with no arguments
        from_config = quiet(instantiate, node)
        from_class = quiet(env_cls)
        assert env_summary(from_config) == env_summary(from_class), f"{name}: instantiated env differs"
        print(f"  {name:20s} defaults OK | yaml OK | instantiate OK "
              f"({from_config.width}x{from_config.height}, max_steps={from_config.max_steps})")


def write_primary_config(conf_dir, body):
    with open(os.path.join(conf_dir, 'config.yaml'), 'w') as f:
        f.write(body)


def test_compose_yaml_searchpath():
    with tempfile.TemporaryDirectory() as conf_dir:
        write_primary_config(conf_dir, """
hydra:
  searchpath:
    - pkg://stable_marl.configs
defaults:
  - env: findgoal
  - wrapper: pettingzoo
  - _self_
wrapper:
  env: ${env}
""")
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            cfg = compose('config', overrides=['env.agents=2', 'env.obs_mode=allo', 'env.tile_size=16',
                                               'env.num_obstacles=6'])
            env = quiet(instantiate, cfg.env)
            obs, _ = quiet(env.reset, seed=0)
            assert env.num_agents == 2 and env.obs_mode == 'allo' and obs[0]['pixels'].shape == (112, 112, 3)
            wrapped = quiet(instantiate, cfg.wrapper)
            assert isinstance(wrapped.env, get_class(cfg.env._target_))
            print(f"  yaml searchpath: env={type(env).__name__} agents={env.num_agents} "
                  f"obs_mode={env.obs_mode} pixels={obs[0]['pixels'].shape} | wrapper={type(wrapped).__name__} OK")

            for name in ENV_CONFIGS:
                cfg = compose('config', overrides=[f'env={name}', 'env.agents=2'])
                env = quiet(instantiate, cfg.env)
                quiet(env.reset, seed=0)
            print(f"  yaml searchpath: all {len(ENV_CONFIGS)} env configs compose + instantiate with agents=2 OK")


def test_compose_config_store():
    register_configs()
    with tempfile.TemporaryDirectory() as conf_dir:
        write_primary_config(conf_dir, """
defaults:
  - env: redbluedoors
  - _self_
""")
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            cfg = compose('config', overrides=['env.agents=3', 'env.size=6'])
            env = quiet(instantiate, cfg.env)
            assert env.num_agents == 3 and (env.width, env.height) == (12, 6)

            # typed: a wrong type is rejected by the dataclass schema
            try:
                compose('config', overrides=['env.agents=three'])
                raise AssertionError("expected a type error")
            except Exception as e:
                assert 'agents' in str(e)
            print(f"  config store: env={type(env).__name__} agents={env.num_agents} "
                  f"grid={env.width}x{env.height} | type checking OK")


if __name__ == '__main__':
    print("defaults / yaml / instantiate:")
    test_defaults_and_yaml()
    print("hydra compose:")
    test_compose_yaml_searchpath()   # before register_configs(), so only the YAML files are used
    test_compose_config_store()
    print("\nAll config tests passed.")
