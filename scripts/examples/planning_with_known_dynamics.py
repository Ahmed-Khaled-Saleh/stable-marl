"""
Model-based planning without learning: plan with the exact navigation dynamics, jointly or per agent.

    python scripts/examples/planning_with_known_dynamics.py

The planning stack is the one used with learned world models (stable-worldmodel's): a `Dynamics`
model rolls candidate action sequences out, an `Objective` scores them (here: distance to each
agent's goal), a solver optimizes them and `WorldModelPolicy` executes the plan (receding horizon).
`mode='joint'` optimizes the sum of the agents' costs over joint plans (centralized);
`mode='per_agent'` lets each agent pick the plan best for its own cost (decentralized).
"""
import stable_marl as sm
from stable_marl.envs.multigrid.dynamics import NavigationDynamics
from stable_marl.planning import CategoricalCEMSolver, GoalMSE, ShootingCostEvaluator
from stable_marl.planning.solver.callbacks import BestCostRecorder, EliteCostRecorder

ENV = dict(agents=3, size=9, num_obstacles=2, n_clutter=0, min_goal_spawn_distance=3)
world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=2, max_episode_steps=40, goal_conditioned=True, **ENV)

world.set_policy(sm.RandomPolicy(seed=0))
print(f"random            success {world.evaluate(episodes=10, seed=0)['success_rate']:5.1f}%")

for mode in ('joint', 'per_agent'):
    cost = ShootingCostEvaluator(NavigationDynamics(), GoalMSE(per_agent=mode == 'per_agent'))
    callbacks = [BestCostRecorder(reduction='none'), EliteCostRecorder()]
    solver = CategoricalCEMSolver(cost, batch_size=2, num_samples=128, n_steps=6, topk=16, mode=mode, seed=0,
                                  callbacks=callbacks)
    policy = sm.WorldModelPolicy(solver, sm.PlanConfig(horizon=8, receding_horizon=2))
    world.set_policy(policy)
    results = world.evaluate(episodes=10, seed=0)
    print(f"CEM, {mode:9s}    success {results['success_rate']:5.1f}%  mean length {results['mean_length']:5.1f}")

    # the callbacks hold the last solve: per CEM iteration, the best cost per env (per agent if 'per_agent')
    best = callbacks[0].history[0]
    print(f"   last solve, best cost per iteration (env 0): {[b[0] for b in best]}")
