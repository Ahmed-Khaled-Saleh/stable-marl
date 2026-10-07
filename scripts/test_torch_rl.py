import gymnasium as gym
import stable_marl.envs
from stable_marl.wrappers.external import TorchRLPettingZooWrapper
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

env = gym.make('MultiGrid-FindGoal-15x15-v0', agents=2, num_obstacles=6)
env = TorchRLPettingZooWrapper(env)   # string agent ids ('agent_0', ...) for TorchRL
env = PettingZooWrapper(
    env=env,
    return_state=False,
    group_map=None,      # default: all agents in one group 'agent'
    use_mask=True,       # agents can finish at different times (success_termination_mode='all')
)
print(env)

# TorchRL works with TensorDicts: run one episode with random actions
env.set_seed(0)
tensordict = env.rollout(max_steps=300)   # stops when the episode is done
print(tensordict)
print("steps:", tensordict.batch_size[0],
      "| total reward per agent:", tensordict["next", "agent", "reward"].sum(0).squeeze(-1).tolist())

env.close()
