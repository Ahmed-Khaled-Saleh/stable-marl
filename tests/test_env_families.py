"""
The RWARE, VMAS and RoboFactory environment families: their example scripts run end to end (VMAS only
with the `vmas` extra, RoboFactory only with ManiSkill, its expert part only with numpy < 2).

    python tests/test_env_families.py
"""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script):
    env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'scripts', 'examples', script)],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-3000:])
    return result.returncode == 0


if __name__ == '__main__':
    scripts = (['rware_warehouse.py'] + (['vmas_scenarios.py'] if importlib.util.find_spec('vmas') else [])
               + (['robofactory_tasks.py'] if importlib.util.find_spec('mani_skill') else []))
    failed = [s for s in scripts if not run(s)]
    print(f'All {len(scripts)} env family examples ran.' if not failed else f'failed: {failed}')
    sys.exit(1 if failed else 0)
