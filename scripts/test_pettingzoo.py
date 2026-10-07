import gymnasium as gym
import multigrid.envs
from multigrid.wrappers.external import PettingZooWrapper

env = gym.make('MultiGrid-Empty-16x16-v0', agents=2, render_mode='human')
env = PettingZooWrapper(env)

observations, infos = env.reset()
while not env.is_done():
   # to be compliant with pettinzoo API, we need to sample actions from the action space of the environment , instead of sampling from the action space of each agent
   actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)
print(terminations)
env.close()