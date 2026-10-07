"""
Run every test of the stable-marl repo.

    python tests/run_all.py                         # all stable-marl tests (no Sionna needed)
    python tests/run_all.py --quick                 # skip the slow RLlib training test
    python tests/run_all.py --sionna-python /path/to/python
        # also export scenes of several envs and check them in Sionna RT with that interpreter
        # (an environment with sionna-rt installed; on CPU, DRJIT_LIBLLVM_PATH may need to point
        #  to LLVM >= 18, it is passed through from the current environment)

Set BLENDER=/path/to/blender to test the .blend export with your Blender (else the bpy module is used).
Exit code 0 if everything passes.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    ('regressions', ['test_regressions.py']),
    ('see-through walls', ['test_see_through_walls.py']),
    ('layout reproducibility', ['test_layout.py']),
    ('hydra configs', ['test_hydra_configs.py']),
    ('external wrappers', ['test_external_wrappers.py']),
    ('sionna export', ['test_sionna_export.py']),
]

# scenes exported for the Sionna RT check: (folder, CLI arguments)
SIONNA_SCENES = [
    ('findgoal', ['--env', 'MultiGrid-FindGoal-15x15-v0', '--agents', '2', '--env-kwargs', 'num_obstacles=6']),
    ('findgoal_low', ['--env', 'MultiGrid-FindGoal-15x15-v0', '--agents', '3', '--seed', '3',
                      '--inner-wall-height', '1.5', '--cell-size', '1']),
    ('redbluedoors_ceiling', ['--env', 'MultiGrid-RedBlueDoors-8x8-v0', '--agents', '2',
                              '--ceiling', 'ceiling_board', '--door-height', '2.2']),
    ('playground', ['--env', 'MultiGrid-Playground-v0', '--agents', '2']),
    ('lockedhallway', ['--env', 'MultiGrid-LockedHallway-4Rooms-v0', '--agents', '2', '--no-center']),
    ('blockedunlockpickup', ['--env', 'MultiGrid-BlockedUnlockPickup-v0', '--agents', '2', '--no-ground']),
]


def run(name, cmd, cwd=None):
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    ok = result.returncode == 0
    last = [l for l in result.stdout.splitlines() if l.strip()][-1:] or ['']
    print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} ({time.time() - start:5.1f} s)  {last[0][:90]}")
    if not ok:
        print('\n'.join('        ' + l for l in (result.stdout + result.stderr).splitlines()[-25:]))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help="skip the RLlib PPO training in the wrapper tests")
    parser.add_argument('--sionna-python', default=None, help="Python interpreter with sionna-rt installed")
    args = parser.parse_args()

    print("stable-marl tests")
    results = []
    for name, script in TESTS:
        extra = ['--skip-rllib-training'] if args.quick and script[0] == 'test_external_wrappers.py' else []
        results.append(run(name, [sys.executable, os.path.join(HERE, script[0]), *extra]))

    if args.sionna_python:
        print("Sionna RT scene checks")
        with tempfile.TemporaryDirectory(prefix='sionna_scenes_') as tmp:
            folders = []
            for folder, cli in SIONNA_SCENES:
                out = os.path.join(tmp, folder)
                ok = run(f"export {folder}", [sys.executable, '-m', 'stable_marl.utils.sionna_export', *cli, '--out', out])
                results.append(ok)
                if ok:
                    folders.append(out)
            results.append(run("check_sionna_scene", [args.sionna_python, os.path.join(HERE, 'check_sionna_scene.py'), *folders]))

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
