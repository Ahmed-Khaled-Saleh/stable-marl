"""
Tests for `stable_marl.utils.sionna_export` that do not need Sionna RT
(the Sionna side is checked by check_sionna_scene.py, see run_all.py --sionna-python).

Checks the command line (`python -m stable_marl.utils.sionna_export`), unique output folders and
overwrite protection, material validation, the grid -> world mapping and headings, the layout
stored in scene.json, the XML / PLY files, and (if a Blender is available) the .blend.

Run:  python tests/test_sionna_export.py            (set BLENDER=/path/to/blender to test the .blend
                                                     with your Blender; else the bpy module is used)
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET

import numpy as np

warnings.filterwarnings('ignore')

import gymnasium as gym
import stable_marl.envs  # noqa: F401
from stable_marl.core.constants import Direction
from stable_marl.utils.sionna_export import (
    ITU_MATERIALS, SceneConfig, check_materials, default_out_dir, export_scene, grid_to_world, world_heading,
)

TESTS = []
TMP = tempfile.mkdtemp(prefix='sionna_export_test_')


def test(fn):
    TESTS.append(fn)
    return fn


def make_env(env_id='MultiGrid-RedBlueDoors-8x8-v0', seed=0, **kwargs):
    env = gym.make(env_id, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=seed)
    return env


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'stable_marl.utils.sionna_export', *args],
                          capture_output=True, text=True, cwd=TMP)


def read_ply(path):
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        counts = dict(line.split()[1:3] for line in header.decode().splitlines() if line.startswith('element'))
        n_vertices, n_faces = int(counts['vertex']), int(counts['face'])
        vertices = np.frombuffer(f.read(12 * n_vertices), dtype='<f4').reshape(-1, 3)
        faces = np.frombuffer(f.read(13 * n_faces), dtype=[('n', 'u1'), ('idx', '<i4', (3,))])
    return vertices, faces


@test
def cli_unique_folders_and_overwrite():
    first = cli('--env', 'MultiGrid-FindGoal-15x15-v0', '--seed', '0')
    second = cli('--env', 'MultiGrid-FindGoal-15x15-v0', '--seed', '1')
    third = cli('--env', 'MultiGrid-FindGoal-15x15-v0', '--seed', '0', '--env-kwargs', 'num_obstacles=6')
    assert first.returncode == second.returncode == third.returncode == 0, first.stderr + second.stderr + third.stderr
    folders = sorted(os.listdir(os.path.join(TMP, 'sionna_scenes')))
    assert len(folders) == 3 and {'FindGoal-15x15_seed0', 'FindGoal-15x15_seed1'} <= set(folders) \
        and any(f.startswith('FindGoal-15x15_seed0_') for f in folders), folders
    again = cli('--env', 'MultiGrid-FindGoal-15x15-v0', '--seed', '0')
    assert again.returncode == 1 and 'already contains a scene' in again.stderr
    assert cli('--env', 'MultiGrid-FindGoal-15x15-v0', '--seed', '0', '--overwrite').returncode == 0


@test
def overwrite_removes_stale_meshes():
    out = os.path.join(TMP, 'same')
    export_scene(make_env('MultiGrid-RedBlueDoors-8x8-v0'), out)
    assert os.path.exists(os.path.join(out, 'meshes', 'door.ply'))
    try:
        export_scene(make_env('MultiGrid-Empty-8x8-v0'), out)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass
    export_scene(make_env('MultiGrid-Empty-8x8-v0'), out, overwrite=True)
    assert sorted(os.listdir(os.path.join(out, 'meshes'))) == ['ground.ply', 'outer_wall.ply']


@test
def default_out_dir_is_unique():
    a = default_out_dir('X', 0, {})
    b = default_out_dir('X', 1, {})
    c = default_out_dir('X', 0, {'num_obstacles': 6})
    d = default_out_dir('X', 0, {'num_obstacles': 7})
    assert len({a, b, c, d}) == 4 and c == default_out_dir('X', 0, {'num_obstacles': 6})


@test
def material_validation():
    check_materials(SceneConfig(), frequency=28e9)                  # defaults valid at 28 GHz
    for itu_type, frequency in (('wet_ground', 28e9), ('not_a_material', None)):
        config = SceneConfig()
        config.materials['ground'] = itu_type
        try:
            check_materials(config, frequency)
            raise AssertionError(f"{itu_type} should be rejected")
        except ValueError:
            pass
    assert len(ITU_MATERIALS) == 19


@test
def mapping_and_headings():
    config = SceneConfig(cell_size=2.0)
    assert grid_to_world(0, 0, 10, 6, config) == (-9.0, 5.0, 0.0)          # top-left cell, grid centred
    assert grid_to_world(9, 5, 10, 6, config) == (9.0, -5.0, 0.0)          # bottom-right cell
    assert grid_to_world(0, 0, 10, 6, SceneConfig(cell_size=1.0, center=False)) == (0.5, -0.5, 0.0)
    expected = {Direction.right: (1, 0), Direction.down: (0, -1), Direction.left: (-1, 0), Direction.up: (0, 1)}
    for direction, (hx, hy) in expected.items():
        h = world_heading(direction)
        assert h['world_heading'][:2] == (hx, hy)
        assert np.isclose(np.cos(h['yaw']), hx) and np.isclose(np.sin(h['yaw']), hy)


@test
def scene_files_and_layout():
    env = make_env('MultiGrid-Playground-v0', agents=2)
    config = SceneConfig(cell_size=1.0, inner_wall_height=1.5)
    out = os.path.join(TMP, 'playground')
    meta = export_scene(env, out, config, frequency=3.5e9)
    u = env.unwrapped

    # scene.xml: one ITU material per id, every shape points to an existing material and mesh
    root = ET.parse(os.path.join(out, 'scene.xml')).getroot()
    materials = {b.get('id'): b for b in root.findall('bsdf')}
    assert all(b.get('type') == 'itu-radio-material' and b.get('id').startswith('mat-itu_') for b in materials.values())
    for shape in root.findall('shape'):
        assert shape.find('ref').get('id') in materials
        assert os.path.exists(os.path.join(out, shape.find('string').get('value')))

    # meshes: triangles only, vertices inside the scene bounds, heights as configured
    half = np.array([u.width, u.height]) * config.cell_size / 2
    for category, info in meta['meshes'].items():
        vertices, faces = read_ply(os.path.join(out, info['file']))
        assert len(faces) == info['triangles'] and (faces['n'] == 3).all()
        assert (np.abs(vertices[:, :2]) <= half + 1e-5).all(), category
    assert np.isclose(read_ply(os.path.join(out, 'meshes', 'inner_wall.ply'))[0][:, 2].max(), 1.5)
    assert np.isclose(read_ply(os.path.join(out, 'meshes', 'outer_wall.ply'))[0][:, 2].max(), 3.0)

    # scene.json: layout matches the env, agents in open cells at antenna height
    layout = meta['layout']
    for x in range(u.width):
        for y in range(u.height):
            obj = u.grid.get(x, y)
            is_wall = obj is not None and obj.type.value == 'wall'
            assert (layout['solid_height'][x][y] > 0) == (is_wall or layout['categories'][x][y] == 'door')
    for agent in meta['agents']:
        x, y = agent['grid_pos']
        assert layout['solid_height'][x][y] == 0
        assert tuple(agent['world_pos']) == grid_to_world(x, y, u.width, u.height, config, config.antenna_height)


@test
def blend_file():
    blender = os.environ.get('BLENDER')
    try:
        if not blender:
            import bpy  # noqa: F401
    except ImportError:
        print("    (skipped: set BLENDER=/path/to/blender or install the bpy module)")
        return
    out = os.path.join(TMP, 'blend')
    export_scene(make_env('MultiGrid-RedBlueDoors-8x8-v0', agents=2), out, blend=True, blender=blender)
    path = os.path.join(out, 'scene.blend')

    # open it again with the same Blender and list its objects / materials
    script = ("import bpy, json; print('OBJECTS', json.dumps({o.name: [m.name for m in getattr(o.data, 'materials', [])] "
              "for o in bpy.data.objects}))")
    if blender:
        cmd = [blender, '-b', path, '--python-expr', script]
    else:
        cmd = [sys.executable, '-c', f"import bpy; bpy.ops.wm.open_mainfile(filepath={path!r}); {script}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in result.stdout.splitlines() if l.startswith('OBJECTS')), None)
    assert line, f"could not open the .blend:\n{result.stdout[-1000:]}{result.stderr[-1000:]}"
    objects = json.loads(line[len('OBJECTS '):])
    assert objects['outer_wall'] == ['itu_concrete'] and objects['door'] == ['itu_wood'], objects
    assert {'agent_0', 'agent_1'} <= set(objects), objects


if __name__ == '__main__':
    import shutil
    try:
        for fn in TESTS:
            fn()
            print(f"  OK  {fn.__name__}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nAll {len(TESTS)} sionna_export tests passed.")
