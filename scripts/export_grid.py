# export_grid.py — run this in your normal Python environment
import json
import numpy as np
from multigrid.envs.findgoal import FindGoalEnv
from multigrid.core.constants import Type

env = FindGoalEnv(
    width=15,
    height=15,
    num_obstacles=6,
    agents=2,           # blue + red agent as in your image
    render_mode='human'
)
obs, info = env.reset(seed=0)

grid_data = {
    "width": env.width,
    "height": env.height,
    "walls": [],
    # "goals": [],
    # "agents": [],
}

# Collect wall/obstacle cells
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell is not None:
            if cell.type == Type.wall:
                grid_data["walls"].append({"x": x, "y": y})
            # elif cell.type == Type.goal:
            #     grid_data["goals"].append({"x": x, "y": y})

# # Collect agent positions
# for agent in env.agents:
#     grid_data["agents"].append({
#         "x": int(agent.state.pos[0]),
#         "y": int(agent.state.pos[1]),
#         "index": agent.index,
#     })

with open("./grid_data.json", "w") as f:
    json.dump(grid_data, f)

print("Grid exported.")