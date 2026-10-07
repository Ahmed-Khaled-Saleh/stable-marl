import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-FindGoal-15x15-v0', agents=2, render_mode='rgb_array', num_obstacles=6, width=15, height=15)


from multigrid.utils.misc import valid_positions


obs, info = env.reset(seed=0)
print("Valid positions:", len(valid_positions(env.unwrapped.grid)))
print("invalid positions:", 13*13 - len(valid_positions(env.unwrapped.grid)))
done = False
while not done:
    actions = {i: env.action_space[i].sample() for i in range(env.unwrapped.num_agents)}
    obs, rewards, terminations, truncations, info = env.step(actions)
    import ipdb; ipdb.set_trace()
    done = all(terminations.values()) or all(truncations.values())

env.close()