"""
Test that `see_through_walls` is respected by both observations:
  * 'image' (symbolic encoding): hidden cells are Type.unseen
  * 'pov'   (RGB rendering):     hidden cells are blacked out

Layout (9x9, agent 0 at (2, 4) facing right, goal hidden behind a wall at x=4):

    W W W W W W W W W
    W . . . W . . . W
    W . . . W . . . W
    W . . . W . . . W
    W . > . W . G . W
    W . . . W . . . W
    W . . . W . . . W
    W . . . W . . . W
    W W W W W W W W W

Run:  python tests/test_see_through_walls.py [--save]
"""
import argparse
import numpy as np

from multigrid.envs.base import MultiGridEnv
from multigrid.core.grid import Grid
from multigrid.core.world_object import Goal
from multigrid.core.constants import Type, Direction

VIEW = 7
GOAL_POS = (6, 4)


class WallTestEnv(MultiGridEnv):
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(4, 1, height - 2)  # interior wall, no gaps
        self.put_obj(Goal(), *GOAL_POS)

        # Deterministic placement: agent 0 faces the wall, agent 1 elsewhere
        self.agents[0].state.pos = (2, 4)
        self.agents[0].state.dir = Direction.right
        self.agents[1].state.pos = (1, 1)
        self.agents[1].state.dir = Direction.down


def make_obs(see_through_walls: bool):
    env = WallTestEnv(agents=2, grid_size=9, agent_view_size=VIEW,
                      see_through_walls=see_through_walls)
    obs, _ = env.reset(seed=0)
    return env, obs[0]


def black_tiles(pov: np.ndarray) -> np.ndarray:
    """(VIEW, VIEW) bool mask, indexed [i, j] like 'image', of fully black POV tiles."""
    ts = pov.shape[0] // VIEW
    return np.array([[pov[j * ts:(j + 1) * ts, i * ts:(i + 1) * ts].max() == 0
                      for j in range(VIEW)] for i in range(VIEW)])


def check(see_through_walls: bool):
    env, obs = make_obs(see_through_walls)
    image, pov = obs['image'], obs['pov']
    types = image[..., 0]
    unseen = types == Type.unseen.to_index()
    goal_seen = (types == Type.goal.to_index()).any()

    # In the agent's view the agent sits at (VIEW//2, VIEW-1) facing "up" (j decreasing).
    # The wall is 2 cells ahead; the goal is 4 cells ahead.
    ci = VIEW // 2
    wall_cell = types[ci, VIEW - 1 - 2]
    behind_wall = unseen[ci, VIEW - 1 - 3]

    print(f"\n--- see_through_walls={see_through_walls} ---")
    print("image types (rows = j, agent at bottom centre):")
    print(types.T)
    print(f"unseen cells: {unseen.sum()} | goal visible: {goal_seen}")

    # The wall right in front is always visible
    assert wall_cell == Type.wall.to_index(), "wall in front should be visible"

    if see_through_walls:
        assert not unseen.any(), "nothing should be unseen when seeing through walls"
        assert goal_seen, "goal behind the wall should be visible"
    else:
        assert behind_wall, "cell right behind the wall should be unseen"
        assert not goal_seen, "goal behind the wall should be hidden"

    # The POV image must hide exactly the cells the encoding marks unseen
    assert (black_tiles(pov) == unseen).all(), "POV blacked-out cells != unseen cells"
    assert env.observation_space[0]['pov'].contains(pov), "POV outside observation space"

    print("OK")
    return env, pov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help="save a side-by-side figure to see_through_walls.png")
    args = parser.parse_args()

    env_f, pov_f = check(see_through_walls=False)
    env_t, pov_t = check(see_through_walls=True)

    if args.save:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(env_f.get_full_render(highlight=True, tile_size=32))
        ax[0].set_title("full grid (highlight = agents' view)")
        ax[1].imshow(pov_f); ax[1].set_title("agent 0 POV, see_through_walls=False")
        ax[2].imshow(pov_t); ax[2].set_title("agent 0 POV, see_through_walls=True")
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig("see_through_walls.png", dpi=150)
        print("\nsaved see_through_walls.png")

    print("\nAll see-through-wall tests passed.")


if __name__ == '__main__':
    main()
