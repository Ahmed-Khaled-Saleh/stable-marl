
import numpy as np

import gymnasium as gym
import multigrid.envs
from multigrid.envs.findgoal import FindGoalEnv
from multigrid.wrappers.external import TorchRLPettingZooWrapper
from torchrl.envs.libs import pettingzoo


# The exact same constants used in your blender_scene.py
CELL_SIZE = 2.0   # metres per grid cell
WALL_H    = 3.0   # only needed if you want the vertical center

def grid_to_world(gx: int, gy: int, cell_size: float = CELL_SIZE):
    """
    Map a 2D grid cell (gx, gy) to the 3D world position (wx, wy, wz).
    
    - wx = gx * cell_size           (grid x → world x, same direction)
    - wy = -gy * cell_size          (grid y → world y, FLIPPED because
                                     Blender Y is forward but grid Y is down)
    - wz = 0.0                      (ground level, agents walk on z=0)
    """
    wx = gx * cell_size
    wy = -gy * cell_size   # flip Y
    wz = 0.0
    return np.array([wx, wy, wz])

def world_to_grid(wx: float, wy: float, cell_size: float = CELL_SIZE):
    """
    Inverse: map a continuous 3D world position back to the nearest grid cell.
    """
    gx = int(round(wx / cell_size))
    gy = int(round(-wy / cell_size))   # un-flip Y
    return np.array([gx, gy])




env = FindGoalEnv(
    width=15,
    height=15,
    num_obstacles=6,
    agents=2,           # blue + red agent as in your image, 
    render_mode= "rgb_array"
)
obs, info = env.reset(seed= 0)
import ipdb; ipdb.set_trace()
import matplotlib.pyplot as plt
plt.imshow(obs["agent_0"]["image"])
plt.axis("off")
plt.show()
CELL_SIZE = 2.0

# Map every agent position to 3D
for agent in env.agents:
    gx, gy = agent.state.pos
    world_pos = grid_to_world(gx, gy)
    print(f"Agent {agent.index}: grid=({gx},{gy})  →  world={world_pos}")
# Map goal position to 3D
# for x in range(env.width):
#     for y in range(env.height):
#         cell = env.grid.get(x, y)
#         if cell is not None and cell.type == Type.goal:
#             print(f"Goal: grid=({x},{y})  →  world={grid_to_world(x, y)}")