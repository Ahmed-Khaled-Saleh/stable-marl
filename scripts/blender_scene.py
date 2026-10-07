# blender_scene.py — paste and run inside Blender > Scripting tab
import bpy
import json
import math

# ── Config ────────────────────────────────────────────────────────────────────
GRID_JSON    = "./grid_data.json"
CELL_SIZE    = 2.0       # metres per grid cell
WALL_H       = 3.0       # building height for outer walls
OBSTACLE_H   = 1.5       # building height for interior obstacles (shorter)
FLOOR_THICK  = 0.1       # floor slab thickness

# Colours (R, G, B)
COL_OUTER_WALL = (0.3, 0.2, 0.1)   # dark brown
COL_OBSTACLE   = (0.55, 0.45, 0.3) # tan/sandstone
COL_FLOOR      = (0.15, 0.15, 0.15)
COL_GOAL       = (0.0, 0.8, 0.1)
COL_AGENT      = [
    (0.1, 0.2, 0.9),   # agent 0 → blue
    (0.9, 0.1, 0.1),   # agent 1 → red
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Also clear orphan materials/meshes
    for block in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights]:
        for item in block:
            block.remove(item)

def make_material(name, color, roughness=0.6, metallic=0.0, emission=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 3.0
    return mat

def add_box(name, location, scale, material):
    """Spawn a UV cube, move and scale it, assign material."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    # Apply scale so dimensions are real
    bpy.ops.object.transform_apply(scale=True)
    obj.data.materials.append(material)
    return obj

def grid_to_world(x, y, z_base, sx, sy, sz):
    """
    Convert grid (x, y) to Blender world coords.
    Blender Y is forward, grid Y is down — so we negate Y.
    z_base is the bottom of the object; we shift up by sz/2 so origin is centre.
    """
    wx = x * CELL_SIZE
    wy = -y * CELL_SIZE   # flip Y
    wz = z_base + sz / 2
    return (wx, wy, wz)

# ── Main ──────────────────────────────────────────────────────────────────────
def build_scene(json_path):
    clear_scene()

    with open(json_path) as f:
        data = json.load(f)

    W, H = data["width"], data["height"]

    # Pre-build materials
    mat_outer    = make_material("M_OuterWall", COL_OUTER_WALL, roughness=0.8)
    mat_obstacle = make_material("M_Obstacle",  COL_OBSTACLE,   roughness=0.7)
    mat_floor    = make_material("M_Floor",     COL_FLOOR,      roughness=0.9)
    mat_goal     = make_material("M_Goal",      COL_GOAL,       roughness=0.3,
                                 emission=COL_GOAL)

    # ── Floor slab ────────────────────────────────────────────────────────────
    fx = (W - 1) * CELL_SIZE / 2
    fy = -(H - 1) * CELL_SIZE / 2
    add_box(
        "Floor",
        location=(fx, fy, -FLOOR_THICK / 2),
        scale=(W * CELL_SIZE, H * CELL_SIZE, FLOOR_THICK),
        material=mat_floor,
    )

    # ── Walls & obstacles ─────────────────────────────────────────────────────
    # Detect border vs interior walls by position
    border_xs = {0, W - 1}
    border_ys = {0, H - 1}

    for i, wall in enumerate(data["walls"]):
        x, y = wall["x"], wall["y"]
        is_border = x in border_xs or y in border_ys
        h      = WALL_H if is_border else OBSTACLE_H
        mat    = mat_outer if is_border else mat_obstacle
        label  = "OuterWall" if is_border else "Obstacle"

        loc = grid_to_world(x, y, 0, CELL_SIZE, CELL_SIZE, h)
        add_box(
            f"{label}_{i}",
            location=loc,
            scale=(CELL_SIZE, CELL_SIZE, h),
            material=mat,
        )

    # # ── Goal ──────────────────────────────────────────────────────────────────
    # for i, goal in enumerate(data["goals"]):
    #     x, y = goal["x"], goal["y"]
    #     # Flat glowing pad on the floor
    #     loc = grid_to_world(x, y, 0, CELL_SIZE, CELL_SIZE, 0.15)
    #     add_box(
    #         f"Goal_{i}",
    #         location=loc,
    #         scale=(CELL_SIZE * 0.8, CELL_SIZE * 0.8, 0.15),
    #         material=mat_goal,
    #     )

    # # ── Agents ────────────────────────────────────────────────────────────────
    # for agent in data["agents"]:
    #     x, y, idx = agent["x"], agent["y"], agent["index"]
    #     mat_agent = make_material(
    #         f"M_Agent{idx}",
    #         COL_AGENT[idx % len(COL_AGENT)],
    #         roughness=0.2,
    #         metallic=0.5,
    #     )
    #     # Agents as cylinders
    #     wx = x * CELL_SIZE
    #     wy = -y * CELL_SIZE
    #     bpy.ops.mesh.primitive_cylinder_add(
    #         radius=CELL_SIZE * 0.3,
    #         depth=1.0,
    #         location=(wx, wy, 0.5),
    #     )
    #     obj = bpy.context.active_object
    #     obj.name = f"Agent_{idx}"
    #     obj.data.materials.append(mat_agent)

    # ── Lighting ──────────────────────────────────────────────────────────────
    # Sun lamp overhead
    bpy.ops.object.light_add(type='SUN', location=(W/2, -H/2, 20))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_x = (W / 2) * CELL_SIZE
    cam_y = -(H / 2) * CELL_SIZE
    bpy.ops.object.camera_add(location=(cam_x, cam_y - H, 25))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(55), 0, 0)
    bpy.context.scene.camera = cam

    print(f"Scene built: {len(data['walls'])} walls, ")
          #f"{len(data['goals'])} goals, {len(data['agents'])} agents.")

build_scene(GRID_JSON)