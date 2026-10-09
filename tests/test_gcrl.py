"""
Tests of MangoBench's decentralized goal-conditioned baselines (stable_marl.wm.gcrl, JAX): the Hydra
pipeline collect -> train/gcrl.py -> plan/eval_ff.py, and each agent acting on its own goal.
Skipped (passing) without the `gcrl` extra.

    python tests/test_gcrl.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn


def script(path, *overrides):
    env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'scripts', path), *overrides],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
    return result.stdout + result.stderr


@test
def hydra_pipeline():
    with tempfile.TemporaryDirectory() as cache:
        script('data/collect_findgoal.py', f'cache_dir={cache}', 'expert_episodes=12', 'random_episodes=4', 'env.tile_size=4')
        for agent in ('gcbc', 'hiql'):
            out = script('train/gcrl.py', f'cache_dir={cache}', f'agent={agent}', 'train.steps=4', 'train.log_every=2',
                         '+agent_config.batch_size=16', *(['+agent_config.subgoal_steps=3'] if agent == 'hiql' else []))
            assert os.path.exists(f'{cache}/checkpoints/{agent}_findgoal/gcrl.pkl'), out
            out = script('plan/eval_ff.py', f'cache_dir={cache}', f'policy={agent}_findgoal', 'eval.episodes=2',
                         'eval.video_episodes=1', 'eval.max_episode_steps=8')
            assert 'success_rate' in out and os.listdir(f'{cache}/checkpoints/{agent}_findgoal/{agent}_findgoal_videos'), out


@test
def agents_act_on_their_own_goal():
    import stable_marl as sm
    from stable_marl.envs.multigrid import GoToGoalPolicy
    from stable_marl.wm.gcrl import IndependentGCRL, agent_datasets
    with tempfile.TemporaryDirectory() as tmp:
        world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=1, goal_conditioned=True, agents=2, size=7,
                         num_obstacles=0, n_clutter=0, tile_size=4, max_steps=15)
        world.set_policy(GoToGoalPolicy())
        world.collect(f'{tmp}/d.h5', episodes=6, seed=0, progress=False)
        data = agent_datasets(sm.HDF5Dataset(path=f'{tmp}/d.h5'))
        model = IndependentGCRL.create('gcbc', data, 4, batch_size=16)
        model.train(data, steps=20, seed=0)
        world.reset(seed=1)
        info = {k: v.copy() for k, v in world.infos.items() if isinstance(v, np.ndarray)}
        base = model.get_action(info)
        # agent 1 gets agent 0's view and goal: agent 0's action is unchanged (its own inputs only)
        info['pixels'][:, :, 1], info['goal'][:, :, 1] = info['pixels'][:, :, 0], info['goal'][:, :, 0]
        assert model.get_action(info)[0, 0] == base[0, 0]


if __name__ == '__main__':
    if importlib.util.find_spec('jax') is None or importlib.util.find_spec('distrax') is None:
        print("All 0 gcrl tests passed (skipped: needs the 'gcrl' extra: jax, flax, distrax).")
        sys.exit(0)
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f'  OK  {fn.__name__}')
        except Exception:
            failed += 1
            print(f'FAIL  {fn.__name__}')
            traceback.print_exc()
    print(f'All {len(TESTS)} gcrl tests passed.' if not failed else f'{failed} / {len(TESTS)} gcrl tests failed.')
    sys.exit(1 if failed else 0)
