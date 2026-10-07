"""
Log the 'ego', 'allo' and 'global' observation modes side by side.

One env is created per mode, all reset with the same seed and stepped with the
same (seeded random) actions, so the trajectories are identical and only the
observation frame differs.

Outputs (in --out):
    step_XXX.png   rows = modes, columns = full grid + each agent's 'pov'
    obs_modes.gif  all steps as an animation
    log.txt        per step: agent states, rewards, terminations, and per mode/agent
                   the obs shapes, visible/unseen cells and the 'image' type grid

Examples:
    python nbs/examples/log_obs_modes.py
    python nbs/examples/log_obs_modes.py --env MultiGrid-FindGoal-15x15-v0 --env-kwargs num_obstacles=6 width=15 height=15
    python nbs/examples/log_obs_modes.py --steps 40 --see-through-walls --out logs/modes
"""
import argparse
import contextlib
import io
import os

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import stable_marl.envs
from stable_marl.envs.multigrid.core.constants import Type

MODES = ('ego', 'allo', 'global')

# Short symbols for the 'image' type grid in the text log
TYPE_TO_STR = {
    Type.unseen: '?', Type.empty: '.', Type.wall: 'W', Type.floor: 'F', Type.door: 'D',
    Type.key: 'K', Type.ball: 'A', Type.box: 'B', Type.goal: 'G', Type.lava: 'V', Type.agent: '@',
}


def parse_kwargs(pairs):
    """Parse ['key=value', ...] into a dict, converting ints / floats / bools."""
    out = {}
    for pair in pairs or []:
        key, value = pair.split('=', 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        else:
            value = {'true': True, 'false': False}.get(value.lower(), value)
        out[key] = value
    return out


def type_grid(image):
    """Render an (w, h, dim) 'image' observation as rows of type symbols (row = y)."""
    rows = []
    for j in range(image.shape[1]):
        row = ''
        for i in range(image.shape[0]):
            row += TYPE_TO_STR.get(Type.from_index(image[i, j, 0]), '*') + ' '
        rows.append(row.rstrip())
    return rows


def make_envs(args):
    envs = {}
    for mode in MODES:
        envs[mode] = gym.make(
            args.env,
            agents=args.agents,
            render_mode='rgb_array',
            obs_mode=mode,
            see_through_walls=args.see_through_walls,
            **parse_kwargs(args.env_kwargs),
        )
    return envs


def quiet(fn, *a, **kw):
    """Run fn while silencing stdout (some envs print debug info)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def save_figure(envs, obs, step, path):
    num_agents = len(obs['ego'])
    fig, axes = plt.subplots(len(MODES), num_agents + 1,
                             figsize=(3.2 * (num_agents + 1), 3.2 * len(MODES)), squeeze=False)
    for r, mode in enumerate(MODES):
        u = envs[mode].unwrapped
        axes[r, 0].imshow(u.get_full_render(highlight=True, tile_size=16))
        axes[r, 0].set_title(f"{mode}: full grid (highlight = view)", fontsize=9)
        for a in range(num_agents):
            color = u.agents[a].state.color.name
            axes[r, a + 1].imshow(obs[mode][a]['pov'])
            axes[r, a + 1].set_title(f"{mode}: agent {a} ({color}) pov", fontsize=9)
    for ax in axes.flat:
        ax.axis('off')
    fig.suptitle(f"step {step}", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def log_step(f, envs, obs, step, actions=None, rewards=None, terms=None, truncs=None):
    ref = envs['ego'].unwrapped  # trajectories are identical across modes
    f.write(f"\n{'=' * 70}\nstep {step}\n{'=' * 70}\n")
    if actions is not None:
        f.write(f"actions: { {k: ref.actions(int(v)).name for k, v in actions.items()} }\n")
        f.write(f"rewards: { {k: round(float(v), 4) for k, v in rewards.items()} }\n")
        f.write(f"terminated: { {k: bool(v) for k, v in terms.items()} } | "
                f"truncated: { {k: bool(v) for k, v in truncs.items()} }\n")
    for agent in ref.agents:
        f.write(f"agent {agent.index} ({agent.state.color.name}): pos={tuple(map(int, agent.state.pos))} "
                f"dir={agent.state.dir.name} terminated={bool(agent.state.terminated)}\n")

    for mode in MODES:
        f.write(f"\n--- obs_mode='{mode}' ---\n")
        for a, o in obs[mode].items():
            image = o['image']
            unseen = int((image[..., 0] == Type.unseen.to_index()).sum())
            f.write(f"agent {a}: image {image.shape} pov {o['pov'].shape} "
                    f"direction={int(o['direction'])} visible={image[..., 0].size - unseen} unseen={unseen}\n")
            for row in type_grid(image):
                f.write(f"    {row}\n")

    # Sanity check: all modes must be on the same trajectory
    for mode in MODES[1:]:
        u = envs[mode].unwrapped
        assert np.array_equal(u.agent_states.pos, ref.agent_states.pos), f"{mode} diverged"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='MultiGrid-RedBlueDoors-8x8-v0')
    parser.add_argument('--env-kwargs', nargs='*', default=[], help="extra env kwargs as key=value")
    parser.add_argument('--agents', type=int, default=2)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--see-through-walls', action='store_true')
    parser.add_argument('--out', default='obs_modes_log')
    parser.add_argument('--fps', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    envs = make_envs(args)
    action_rng = np.random.default_rng(args.seed)
    frames = []

    with open(os.path.join(args.out, 'log.txt'), 'w') as f:
        f.write(f"env={args.env} agents={args.agents} seed={args.seed} "
                f"see_through_walls={args.see_through_walls} kwargs={parse_kwargs(args.env_kwargs)}\n")
        f.write("legend: ? unseen  . empty  W wall  D door  K key  A ball  B box  G goal  V lava  @ agent\n")
        f.write("image grids are printed with row = y (top to bottom), column = x (left to right)\n")

        obs = {mode: quiet(envs[mode].reset, seed=args.seed)[0] for mode in MODES}
        log_step(f, envs, obs, step=0)
        path = os.path.join(args.out, 'step_000.png')
        save_figure(envs, obs, 0, path)
        frames.append(path)

        for step in range(1, args.steps + 1):
            ref = envs['ego'].unwrapped
            actions = {a.index: int(action_rng.integers(ref.action_space[a.index].n)) for a in ref.agents}
            results = {mode: quiet(envs[mode].step, actions) for mode in MODES}
            obs = {mode: results[mode][0] for mode in MODES}
            _, rewards, terms, truncs, _ = results['ego']

            log_step(f, envs, obs, step, actions, rewards, terms, truncs)
            path = os.path.join(args.out, f'step_{step:03d}.png')
            save_figure(envs, obs, step, path)
            frames.append(path)
            print(f"step {step:3d} | rewards {dict(rewards)} | terminated {[bool(t) for t in terms.values()]}")

            if all(terms.values()) or all(truncs.values()):
                print("episode finished")
                break

    images = [Image.open(p).convert('RGB') for p in frames]
    images[0].save(os.path.join(args.out, 'obs_modes.gif'), save_all=True,
                   append_images=images[1:], duration=int(1000 / args.fps), loop=0)
    print(f"\nsaved {len(frames)} step images, obs_modes.gif and log.txt to {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
