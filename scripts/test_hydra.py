# train.py
from build import env
import hydra
from omegaconf import DictConfig
from multigrid.core import actions
from multigrid.utils.env_factory import make_env

@hydra.main(config_path="env", config_name="playground")
def main(cfg: DictConfig):
    env = make_env(cfg)
    observations, infos = env.reset()
    while not env.unwrapped.is_done():
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
   
    env.close()


if __name__ == "__main__":
    main()
