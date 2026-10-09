"""
Communication between agents (stable_marl.comm, with comm-core): the example script runs end to end.
Skipped without a comm-core that has networks.

    python tests/test_comm.py
"""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    if importlib.util.find_spec('comm_core') is None or importlib.util.find_spec('comm_core.network') is None:
        print('comm-core (with networks) not installed: skipped')
        sys.exit(0)
    env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(ROOT, 'scripts', 'examples', 'communication.py')],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-3000:])
        sys.exit(1)
    print('Communication example ran.')
