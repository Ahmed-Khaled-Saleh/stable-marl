"""
Check that a scene exported by `multigrid.utils.sionna_export` matches its MultiGrid layout in Sionna RT,
and map positions between the two worlds.

Run it in the Python environment that has Sionna RT (multigrid is not needed):

    python tests/check_sionna_scene.py sionna_scenes/FindGoal-15x15_seed0 [more scene folders ...]

Checks (by loading scene.xml with `sionna.rt.load_scene` and casting rays):
  1. materials    every exported category is an ITU radio material with the expected type / thickness
  2. cells        a vertical ray onto every grid cell hits at that cell's height (wall, door, object, ground)
  3. walls        from every open cell, a horizontal ray towards a solid neighbour hits it exactly at the
                  cell edge with the face pointing back, and nothing blocks the way to an open neighbour
  4. agents       each agent sits in its grid cell (an open cell) at antenna height, heading == yaw

Exit code 0 if every scene passes, 1 otherwise.

Using the mapping in your own code (e.g. to move transmitters / receivers as agents move):

    from check_sionna_scene import GridMapping
    m = GridMapping.from_scene('sionna_scenes/FindGoal-15x15_seed0')
    tx.position = m.to_world(*agent.state.pos, z=m.antenna_height)
    tx.orientation = [m.yaw(agent.state.dir), 0, 0]
    x, y = m.to_grid(*rx.position[:2])
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# grid step (dx, dy) -> world direction; grid y points down, world y points up
GRID_DIRECTIONS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}   # right, down, left, up
SOLID = ('outer_wall', 'inner_wall', 'door')


class GridMapping:
    """Grid <-> Sionna world coordinates of an exported scene (see scene.json 'mapping')."""

    def __init__(self, origin, cell_size, width, height, antenna_height=1.5):
        self.origin = np.asarray(origin, dtype=float)   # world position of the grid's top-left corner
        self.cell_size = float(cell_size)
        self.width, self.height = int(width), int(height)
        self.antenna_height = float(antenna_height)

    @classmethod
    def from_scene(cls, scene_dir: str) -> 'GridMapping':
        with open(os.path.join(scene_dir, 'scene.json')) as f:
            meta = json.load(f)
        return cls(meta['mapping']['origin'], meta['mapping']['cell_size'],
                   meta['grid']['width'], meta['grid']['height'], meta['config']['antenna_height'])

    def to_world(self, x: float, y: float, z: float = 0.0) -> list[float]:
        """World position (metres) of the centre of grid cell (x, y)."""
        cs = self.cell_size
        return [float(self.origin[0] + (x + 0.5) * cs), float(self.origin[1] - (y + 0.5) * cs), float(z)]

    def to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Grid cell containing world position (wx, wy)."""
        cs = self.cell_size
        return int(math.floor((wx - self.origin[0]) / cs)), int(math.floor((self.origin[1] - wy) / cs))

    @staticmethod
    def heading(direction: int) -> list[float]:
        """World unit vector of a MultiGrid direction (0 right, 1 down, 2 left, 3 up)."""
        dx, dy = GRID_DIRECTIONS[int(direction)]
        return [float(dx), float(-dy), 0.0]

    @classmethod
    def yaw(cls, direction: int) -> float:
        """Sionna yaw (radians, counter-clockwise from +x) of a MultiGrid direction."""
        hx, hy, _ = cls.heading(direction)
        return math.atan2(hy, hx)


def check_scene(scene_dir: str, tol: float = 1e-4) -> list[str]:
    """Run all checks on an exported scene folder; returns the list of failures (empty = pass)."""
    import mitsuba as mi
    from sionna.rt import ITURadioMaterial, load_scene

    with open(os.path.join(scene_dir, 'scene.json')) as f:
        meta = json.load(f)
    if 'layout' not in meta:
        return ["scene.json has no 'layout' (re-export the scene with the current multigrid.utils.sionna_export)"]
    scene = load_scene(os.path.join(scene_dir, 'scene.xml'), merge_shapes=False)
    mapping = GridMapping.from_scene(scene_dir)
    W, H, cs = mapping.width, mapping.height, mapping.cell_size
    categories = meta['layout']['categories']
    solid, surface = meta['layout']['solid_height'], meta['layout']['surface_height']
    has_ground = 'ground' in meta['meshes']
    failures = []

    def cast(origins, directions):
        o, d = np.asarray(origins, dtype=float), np.asarray(directions, dtype=float)
        si = scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(*o.T), d=mi.Vector3f(*d.T)))
        return np.array(si.t), np.array(si.n).T

    # 1. materials
    for category, info in meta['materials'].items():
        name = category if category in scene.objects else f'mesh-{category}'
        if name not in scene.objects:
            failures.append(f"materials: no object for '{category}'")
            continue
        material = scene.objects[name].radio_material
        thickness = float(np.array(material.thickness).ravel()[0])
        if not isinstance(material, ITURadioMaterial) or material.itu_type != info['itu_type']:
            found = getattr(material, 'itu_type', type(material).__name__)
            failures.append(f"materials: '{category}' is '{found}', expected ITU '{info['itu_type']}'")
        elif abs(thickness - info['thickness']) > 1e-5:
            failures.append(f"materials: '{category}' thickness {thickness} != {info['thickness']}")

    # 2. cells: vertical rays onto every cell centre
    top = 10.0 + max(max(col) for col in surface)
    cells = [(x, y) for x in range(W) for y in range(H)]
    t, n = cast([mapping.to_world(x, y, top) for x, y in cells], [(0, 0, -1)] * len(cells))
    ceiling = max(meta['config']['outer_wall_height'], meta['config']['inner_wall_height'])
    if 'ceiling' in meta['meshes']:
        # rays from above hit the ceiling first: continue below it where the cell is lower
        below = [i for i, (x, y) in enumerate(cells)
                 if abs((top - t[i]) - ceiling) < 1e-3 and abs(surface[x][y] - ceiling) > 1e-3]
        if below:
            t2, n2 = cast([mapping.to_world(*cells[i], ceiling - 1e-3) for i in below], [(0, 0, -1)] * len(below))
            for i, t_i, n_i in zip(below, t2, n2):
                t[i], n[i] = t_i + (top - ceiling + 1e-3), n_i
    for (x, y), t_hit, normal in zip(cells, t, n):
        expected = surface[x][y]
        hit = np.isfinite(t_hit) and t_hit < 1e30
        if expected == 0 and not has_ground:
            if hit:
                failures.append(f"cells: unexpected surface at grid {(x, y)} (z={top - t_hit:.3f})")
            continue
        if not hit or abs((top - t_hit) - expected) > 1e-3:
            got = f"{top - t_hit:.3f}" if hit else "nothing"
            failures.append(f"cells: grid {(x, y)} [{categories[x][y]}] expected surface at z={expected}, hit {got}")
        elif 'ceiling' in meta['meshes'] and abs(expected - ceiling) < 1e-3:
            if abs(normal[2] + 1) > tol:        # covered by the ceiling (facing down)
                failures.append(f"cells: grid {(x, y)} at ceiling height, normal {normal.round(3).tolist()} is not -z")
        elif categories[x][y] != 'ball' and abs(normal[2] - 1) > tol:
            failures.append(f"cells: grid {(x, y)} top face normal {normal.round(3).tolist()} is not +z")

    # 3. walls: horizontal rays from open cells to their 4 neighbours
    solid_heights = [h for col in solid for h in col if h > 0]
    if solid_heights:
        z = 0.5 * min(solid_heights)
        origins, directions, neighbours = [], [], []
        for x, y in cells:
            if solid[x][y] > 0:
                continue
            for direction, (dx, dy) in GRID_DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    origins.append(mapping.to_world(x, y, z))
                    directions.append(mapping.heading(direction))
                    neighbours.append((x, y, nx, ny))
        t, n = cast(origins, directions)
        for (x, y, nx, ny), t_hit, normal, d in zip(neighbours, t, n, directions):
            if solid[nx][ny] > z:
                if abs(t_hit - cs / 2) > tol:
                    failures.append(f"walls: from {(x, y)}, solid {(nx, ny)} hit at {t_hit:.4f} m, expected {cs / 2}")
                elif abs(np.dot(normal, d) + 1) > tol:
                    failures.append(f"walls: face of {(nx, ny)} seen from {(x, y)} has normal {normal.round(3).tolist()}")
            elif t_hit < cs / 2 - tol:
                failures.append(f"walls: way from {(x, y)} to open {(nx, ny)} blocked at {t_hit:.4f} m")

    # 4. agents
    for agent in meta['agents']:
        wx, wy, wz = agent['world_pos']
        gx, gy = mapping.to_grid(wx, wy)
        if [gx, gy] != agent['grid_pos']:
            failures.append(f"agents: agent {agent['index']} world {agent['world_pos']} maps to {(gx, gy)}, "
                            f"expected {agent['grid_pos']}")
        elif solid[gx][gy] > 0:
            failures.append(f"agents: agent {agent['index']} is inside solid cell {(gx, gy)}")
        if abs(wz - mapping.antenna_height) > tol:
            failures.append(f"agents: agent {agent['index']} at z={wz}, expected {mapping.antenna_height}")
        if not np.allclose(agent['world_heading'], mapping.heading(agent['dir'])) \
                or abs(agent['yaw'] - mapping.yaw(agent['dir'])) > tol:
            failures.append(f"agents: agent {agent['index']} heading / yaw inconsistent with direction {agent['dir']}")

    print(f"{scene_dir}: {meta['source'].get('env')} {W}x{H}, {len(meta['materials'])} materials, "
          f"{len(cells)} cells, {len(meta['agents'])} agents -> "
          + ("OK" if not failures else f"{len(failures)} FAILURES"))
    for failure in failures[:20]:
        print("   ", failure)
    return failures


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [check_scene(scene_dir) for scene_dir in sys.argv[1:]]
    sys.exit(0 if all(not r for r in results) else 1)
