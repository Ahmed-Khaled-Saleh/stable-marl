import gymnasium as gym
import multigrid.envs
import matplotlib.pyplot as plt


env_kwargs= {"allow_agent_overlap": False, "success_termination_mode": "all"}
env = gym.make('MultiGrid-Empty-Random-5x5-v0', agents=2, render_mode='human', **env_kwargs)

def plot_two_imgs_in_one(img1, img2):
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.savefig('two_imgs.png')
    plt.show()

goal_obs = None
observations, infos = env.reset()
while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # if goal_obs is None:
    #     goal_obs = env.unwrapped.get_goal_state(
    #         env.unwrapped.agents[1],
    #         env.unwrapped.agents[0].view_size
    #     )
    # if env.unwrapped.step_count % 5 == 0:
    #     img1 = env.render()
    #     # img2 = observations[0]['pov']
    #     plot_two_imgs_in_one(img1, goal_obs)
    #     break
    
    # plt.imshow(goal_obs)
    # plt.show()
    # break

env.close()