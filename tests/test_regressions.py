"""
Regression tests for the bugs found in the codebase audit (each one was reproduced before fixing).

Run:  python tests/test_regressions.py
"""
import contextlib
import io
import tempfile
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from stable_marl.core.actions import Action
from stable_marl.core.constants import Color, Direction, Type
from stable_marl.core.grid import Grid
from stable_marl.core.world_object import Ball, Box, Door, Key, Marker, Wall, WorldObj
from stable_marl.envs import (
    BlockedUnlockPickupEnv, EmptyEnv, LockedHallwayEnv, PlaygroundEnv, RedBlueDoorsEnv,
)

F, L, TOGGLE, PICKUP = int(Action.forward), int(Action.left), int(Action.toggle), int(Action.pickup)
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def where(env, obj):
    return next(pos for pos, o in list(env.grid.world_objects.items()) if o is obj)


def face(env, agent, target_pos):
    """Put `agent` on a free cell next to `target_pos`, facing it."""
    tx, ty = target_pos
    for d in Direction:
        dx, dy = d.to_vec()
        cell = (tx - dx, ty - dy)
        if 0 <= cell[0] < env.width and 0 <= cell[1] < env.height and env.grid.get(*cell) is None:
            agent.state.pos, agent.state.dir = cell, d
            return
    raise RuntimeError(f"no free cell next to {target_pos}")


# ------------------------------------------------------------------ envs
@test
def door_cannot_close_on_agent():
    env = RedBlueDoorsEnv(agents=2)
    env.reset(seed=0)
    door = env.red_door
    x, y = where(env, door)
    door.is_open = True
    env.grid.update(x, y)
    env.agents[1].state.pos = (x, y)                       # agent 1 stands in the open doorway
    env.agents[0].state.pos, env.agents[0].state.dir = (x + 1, y), Direction.left
    env.step({0: TOGGLE})
    assert door.is_open, "door closed on an agent"
    env.agents[1].state.pos = (x + 1, y - 1)               # doorway free again: closing works
    env.step({0: TOGGLE})
    assert not door.is_open


@test
def locked_hallway_completion_terminates_agents():
    env = LockedHallwayEnv(num_rooms=2, agents=2)
    env.reset(seed=0)
    agent = env.agents[0]
    total = 0
    for door in env.locked_doors:
        agent.state.carrying = Key(color=door.color)
        face(env, agent, where(env, door))
        _, rewards, terms, _, _ = env.step({0: TOGGLE, 1: L})
        total += rewards[0]
    assert all(terms.values()) and env.is_done(), f"terminations={terms}, is_done={env.is_done()}"
    assert all(a.state.terminated for a in env.agents)
    assert total > 0 and len(env.unlocked_doors) == len(env.locked_doors) == 2


@test
def locked_hallway_more_rooms_than_colors():
    env = LockedHallwayEnv(num_rooms=8, agents=1)
    env.reset(seed=0)
    assert len(env.locked_doors) == 8, len(env.locked_doors)


@test
def blocked_unlock_pickup_mission():
    for seed in range(5):
        env = BlockedUnlockPickupEnv(agents=2)
        obs, _ = env.reset(seed=seed)
        expected = f"pick up the {env.obj.color.value} {env.obj.type.value}"
        assert str(env.mission) == expected, (str(env.mission), expected)
        assert all(str(o['mission']) == expected for o in obs.values())
        assert env.mission_space.contains(env.mission)


@test
def blocked_unlock_pickup_opening_target_box_fails():
    env = BlockedUnlockPickupEnv(agents=2)
    env.reset(seed=0)
    face(env, env.agents[0], where(env, env.obj))
    _, rewards, terms, _, _ = env.step({0: TOGGLE, 1: L})
    assert all(terms.values()) and env.is_done() and sum(rewards.values()) == 0


@test
def blocked_unlock_pickup_success_still_works():
    env = BlockedUnlockPickupEnv(agents=2)
    env.reset(seed=0)
    face(env, env.agents[0], where(env, env.obj))
    _, rewards, terms, _, _ = env.step({0: PICKUP, 1: L})
    assert all(terms.values()) and rewards[0] > 0


@test
def empty_no_overlap_start_positions():
    env = EmptyEnv(agents=3, allow_agent_overlap=False)
    env.reset(seed=0)
    positions = [tuple(map(int, a.state.pos)) for a in env.agents]
    assert len(set(positions)) == 3 and positions[0] == (1, 1), positions


@test
def str_with_marker():
    env = EmptyEnv(agents=1)
    env.reset(seed=0)
    env.put_obj(Marker(), 2, 2)
    assert 'M' in str(env)


@test
def envs_export_classes():
    import stable_marl.envs as envs
    assert {'EmptyEnv', 'FindGoalEnv', 'RedBlueDoorsEnv'} <= set(envs.__all__)


# ------------------------------------------------------------------ room grid
@test
def add_door_random_direction():
    env = PlaygroundEnv(agents=1)
    env.reset(seed=0)
    for col in range(env.num_cols):
        for row in range(env.num_rows):
            room = env.get_room(col, row)
            if any(room.neighbors[d] is not None and room.doors[d] is None for d in Direction):
                door, _ = env.add_door(col, row)
                assert isinstance(door, Door)
                return
    raise AssertionError("no room with a free wall")


@test
def room_locked_after_remove_wall():
    env = PlaygroundEnv(agents=1)
    env.reset(seed=0)
    for col in range(env.num_cols):
        for row in range(env.num_rows):
            room = env.get_room(col, row)
            for d in Direction:
                if room.neighbors[d] is not None and room.doors[d] is None:
                    env.remove_wall(col, row, d)
                    assert room.locked in (True, False)     # used to raise AttributeError
                    return
    raise AssertionError("no removable wall")


@test
def add_distractors():
    env = PlaygroundEnv(agents=1)
    env.reset(seed=0)
    distractors = env.add_distractors(num_distractors=5)
    assert len(distractors) == 5
    rooms = {(env.room_from_pos(*o.cur_pos).top) for o in distractors}
    assert len(rooms) > 1, "all distractors were placed in the same room"
    try:                                       # more unique distractors than combinations left
        env.add_distractors(num_distractors=50)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ------------------------------------------------------------------ grid / world objects
@test
def walls_drawn_after_get():
    grid = Grid(5, 5)
    grid.get(2, 2)
    grid.vert_wall(2, 0)
    grid.get(1, 1)
    grid.horz_wall(0, 1)
    assert grid.get(2, 2).type == Type.wall and grid.get(1, 1).type == Type.wall


@test
def grid_encode_vis_mask():
    grid = Grid(4, 4)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    encoding = grid.encode(mask)
    assert (encoding[..., 0] == Type.unseen.to_index()).sum() == 15


@test
def decoding_does_not_mutate_cached_wall():
    WorldObj.from_array(np.array([Type.wall.to_index(), Color.red.to_index(), 0]))
    assert Wall().color == Color.grey


@test
def box_contents_get_position():
    env = EmptyEnv(agents=1)
    env.reset(seed=0)
    ball = Ball()
    env.put_obj(Box(contains=ball), 3, 3)
    env.agents[0].state.pos, env.agents[0].state.dir = (2, 3), Direction.right
    env.step({0: TOGGLE})
    assert env.grid.get(3, 3) is ball and tuple(ball.cur_pos) == (3, 3)


# ------------------------------------------------------------------ wrappers
@test
def grid_recorder_long_episode():
    from stable_marl.wrappers.base import GridRecorder
    env = GridRecorder(EmptyEnv(size=16, agents=1, render_mode='rgb_array'), save_root=tempfile.mkdtemp(),
                       auto_save_images=False, auto_save_videos=False)
    env.recording, env.video_scale = True, 1
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, _, truncs, _ = env.step({0: L})
        steps += 1
        if all(truncs.values()):
            break
    assert steps == 1024 and env.ptr == steps + 1, (steps, env.ptr)


@test
def fully_obs_wrapper_shape():
    from stable_marl.wrappers.base import FullyObsWrapper
    env = FullyObsWrapper(RedBlueDoorsEnv(agents=2))
    obs, _ = env.reset(seed=0)
    space = env.unwrapped.agents[0].observation_space['image']
    assert obs[0]['image'].shape == space.shape == (16, 8, 3) and space.contains(obs[0]['image'])


if __name__ == '__main__':
    for fn in TESTS:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        print(f"  OK  {fn.__name__}")
    print(f"\nAll {len(TESTS)} regression tests passed.")
