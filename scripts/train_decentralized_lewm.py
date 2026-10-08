"""
Decentralized LeWMs on a small FindGoal (random goals and starts, 2 agents by default): collect expert + random data,
train one LeWM per agent on its own local view, then let each agent plan with its own model.

    python scripts/train_decentralized_lewm.py [--epochs 20] [--obs-mode allo] [--position] [--agents 2]

`--position` adds each agent's own position as an extra model input (with the goal's position for
the goal), `--direction` its facing direction; `--image-size` sets the model's input size. The trained model is evaluated with the goal scored at the last predicted step
('last', as in stable-worldmodel) and at the plan's best step ('min').

Small model and images so it runs on CPU (about 5 minutes for 20 epochs); use the default
`build_lewm` configuration (ViT-tiny, 224 x 224) on a GPU.
"""
import argparse
import tempfile
import time

import torch

import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_marl.wm import DecentralizedWorldModel, train_world_model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--obs-mode', default='allo', choices=['ego', 'allo', 'global'],
                    help="'ego': local view rotating with the agent, 'allo': world-aligned local view, 'global': whole grid")
parser.add_argument('--position', action='store_true', help="add each agent's position as a model input")
parser.add_argument('--direction', action='store_true', help="add each agent's facing direction as a model input")
parser.add_argument('--image-size', type=int, default=32, help='model input size (pov is resized to it)')
parser.add_argument('--agents', type=int, default=2)
parser.add_argument('--episodes', type=int, default=20, help='evaluation episodes')
args = parser.parse_args()

ENV = dict(agents=args.agents, tile_size=8, size=7, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1, obs_mode=args.obs_mode)
SMALL = dict(image_size=args.image_size, patch_size=8, embed_dim=64, depth=3, heads=4, dim_head=16, mlp_dim=256, projector_hidden=256,
             encoder_kwargs=dict(dim=64, depth=3, heads=4, mlp_dim=256), history_size=3, dropout=0.0)
extras = {**({'position': 2} if args.position else {}), **({'direction': 1} if args.direction else {})}
if extras:
    SMALL['extra_inputs'] = extras
inputs = ['pov', *extras]
torch.manual_seed(0)

tmp = tempfile.mkdtemp()
t = time.time()
for policy, episodes, seed in ((GoToGoalPolicy(), 300, 0), (sm.RandomPolicy(seed=0), 150, 50_000)):
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, max_episode_steps=25, goal_conditioned=True, **ENV)
    world.set_policy(policy)
    world.collect(f'{tmp}/data.h5', episodes=episodes, seed=seed, progress=False)
ds = sm.HDF5Dataset(path=f'{tmp}/data.h5', num_steps=4, keys_to_load=[*inputs, 'action', 'terminated'])
print(f'agents={args.agents} obs_mode={args.obs_mode} image_size={args.image_size} inputs={inputs}: {len(ds.lengths)} episodes, {len(ds)} clips ({time.time() - t:.0f}s)')

model = DecentralizedWorldModel.build_lewm(num_agents=args.agents, num_actions=4, **SMALL)
t = time.time()
hist = train_world_model(model, ds, epochs=args.epochs, batch_size=64, lr=1e-3, sigreg_kwargs={'num_proj': 256}, log=None)
print(f'trained {args.epochs} epochs in {time.time() - t:.0f}s: train {hist["train_loss"][0]:.3f} -> '
      f'{hist["train_loss"][-1]:.3f}, val {hist["val_loss"][0]:.3f} -> {hist["val_loss"][-1]:.3f}')


def planner(step_reduction):
    cost = ShootingCostEvaluator(model, GoalMSE(per_agent=True, step_reduction=step_reduction))
    solver = CategoricalCEMSolver(cost, batch_size=args.agents, num_samples=128, n_steps=5, topk=16, mode='per_agent', seed=0)
    return sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2, history_len=3), history_keys=inputs)


for name, policy in (('random', sm.RandomPolicy(seed=1)), ("LeWM, goal at the last step", planner('last')),
                     ("LeWM, goal at the best step", planner('min')), ('expert', GoToGoalPolicy())):
    t = time.time()
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=40, goal_conditioned=True, **ENV)
    world.set_policy(policy)
    r = world.evaluate(episodes=args.episodes, seed=100_000)
    print(f'{name:28s} success {r["success_rate"]:5.1f}%  mean length {r["mean_length"]:5.1f}  ({time.time() - t:.0f}s)')
