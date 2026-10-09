"""
Tests of MAMBA on MultiGrid (stable_marl.wm.mamba): the Hydra pipeline train/mamba.py -> plan/eval_ff.py
for MAMBA (shaped reward) and GC-MAMBA (goal offsets, hindsight relabelling), with a tiny model.

    python tests/test_mamba.py
"""
import os
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = []
def test(fn):
    TESTS.append(fn)
    return fn

TINY = [f'+mamba_config.{k}={v}' for k, v in dict(HIDDEN=32, MODEL_HIDDEN=32, EMBED=32, N_CATEGORICALS=4, N_CLASSES=4,
                                                    DETERMINISTIC=32, VALUE_HIDDEN=32, PCONT_HIDDEN=32, ACTION_HIDDEN=32,
                                                    REWARD_HIDDEN=32, MIN_BUFFER_SIZE=40, SEQ_LENGTH=10, BATCH_SIZE=4,
                                                    MODEL_BATCH_SIZE=4, PPO_EPOCHS=1).items()]


def script(path, *overrides):
    env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'scripts', path), *overrides],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
    return result.stdout + result.stderr


@test
def hydra_pipeline():
    with tempfile.TemporaryDirectory() as cache:
        for gc, name in (('false', 'mamba_findgoal'), ('true', 'gc_mamba_findgoal')):
            script('train/mamba.py', f'cache_dir={cache}', f'goal_conditioned={gc}', 'train.episodes=5', 'train.log_every=5',
                   'env.size=7', 'env.num_obstacles=0', 'env.max_steps=15', *TINY)
            assert os.path.exists(f'{cache}/checkpoints/{name}/mamba.pt')
            out = script('plan/eval_ff.py', f'cache_dir={cache}', f'policy={name}', 'eval.episodes=2', 'eval.video_episodes=0',
                         'eval.max_episode_steps=8')
            assert 'success_rate' in out, out


if __name__ == '__main__':
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f'  OK  {fn.__name__}')
        except Exception:
            failed += 1
            print(f'FAIL  {fn.__name__}')
            traceback.print_exc()
    print(f'All {len(TESTS)} mamba tests passed.' if not failed else f'{failed} / {len(TESTS)} mamba tests failed.')
    sys.exit(1 if failed else 0)
