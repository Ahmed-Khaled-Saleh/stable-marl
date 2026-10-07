"""
Decentralized LeWMs on a small FindGoal (random goals and starts, 2 agents): collect expert + random data,
train one LeWM per agent on its own pov, then let each agent plan with its own model.

    python scripts/train_decentralized_lewm.py [epochs] [ego|allo]

Small model and images so it runs on CPU (about 5 minutes for 20 epochs); use the default
`build_lewm` configuration (ViT-tiny, 224 x 224) on a GPU.
"""
import sys, tempfile, time
import numpy as np, torch
import stable_marl as sm
from stable_marl.envs.multigrid import GoToGoalPolicy
from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_marl.wm import DecentralizedWorldModel, train_world_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OBS_MODE = sys.argv[2] if len(sys.argv) > 2 else 'allo'   # 'ego' (rotates with the agent) or 'allo' (world-aligned)
ENV = dict(agents=2, tile_size=8, size=7, num_obstacles=0, n_clutter=0, min_goal_spawn_distance=1, obs_mode=OBS_MODE)
SMALL = dict(image_size=32, patch_size=8, embed_dim=64, depth=3, heads=4, dim_head=16, mlp_dim=256, projector_hidden=256,
             encoder_kwargs=dict(dim=64, depth=3, heads=4, mlp_dim=256), history_size=3, dropout=0.0)
torch.manual_seed(0)
tmp = tempfile.mkdtemp()
t = time.time()
for name, policy, episodes in (('expert', GoToGoalPolicy(), 300), ('random', sm.RandomPolicy(seed=0), 150)):
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, max_episode_steps=25, goal_conditioned=True, **ENV)
    world.set_policy(policy)
    world.collect(f'{tmp}/data.h5', episodes=episodes, seed=0 if name == 'expert' else 50_000, progress=False)
ds = sm.HDF5Dataset(path=f'{tmp}/data.h5', num_steps=4, keys_to_load=['pov', 'action', 'terminated'])
print(f'obs_mode={OBS_MODE} dataset: {len(ds.lengths)} episodes, {len(ds)} clips ({time.time() - t:.0f}s)')

model = DecentralizedWorldModel.build_lewm(num_agents=2, num_actions=4, **SMALL)
untrained = DecentralizedWorldModel.build_lewm(num_agents=2, num_actions=4, **SMALL)
t = time.time()
hist = train_world_model(model, ds, epochs=EPOCHS, batch_size=64, lr=1e-3, sigreg_kwargs={'num_proj': 256}, log=None)
print(f'trained {EPOCHS} epochs in {time.time() - t:.0f}s: train {hist["train_loss"][0]:.3f} -> {hist["train_loss"][-1]:.3f}, '
      f'val {hist["val_loss"][0]:.3f} -> {hist["val_loss"][-1]:.3f}')

# prediction quality vs "nothing changes" (copy the current embedding)
model.eval()
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(1))
batch = next(iter(loader))
with torch.no_grad():
    errs = {'model': [], 'copy': []}
    for a, wm in enumerate(model.models):
        B, T = batch['pov'].shape[:2]
        acts = model.agent_actions(batch['action'].reshape(B, T, 1, 2, -1)[:, :, :, a].reshape(B, T, -1))
        enc = wm.encode({'pixels': batch['pov'][:, :, a], 'action': acts})
        emb = enc['emb']
        pred = wm.predict(emb[:, :3], enc['act_emb'][:, :3])
        errs['model'].append((pred - emb[:, 1:4]).pow(2).mean().item())
        errs['copy'].append((emb[:, :3] - emb[:, 1:4]).pow(2).mean().item())
print('next-embedding error: model', [round(e, 3) for e in errs['model']], ' copy-current baseline', [round(e, 3) for e in errs['copy']])

def planner(wm):
    cost = ShootingCostEvaluator(wm, GoalMSE(per_agent=True))
    solver = CategoricalCEMSolver(cost, batch_size=2, num_samples=128, n_steps=5, topk=16, mode='per_agent', seed=0)
    return sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2, history_len=3))

for name, policy in (('random', sm.RandomPolicy(seed=1)), ('untrained LeWM', planner(untrained)),
                     ('trained LeWM', planner(model)), ('expert', GoToGoalPolicy())):
    t = time.time()
    world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=40, goal_conditioned=True, **ENV)
    world.set_policy(policy)
    r = world.evaluate(episodes=20, seed=100_000)
    print(f'{name:15s} success {r["success_rate"]:5.1f}%  mean length {r["mean_length"]:5.1f}  ({time.time() - t:.0f}s)')
