"""
Agents that communicate, with comm-core modelling the communication (pip install 'stable-marl[comm]').

    python scripts/examples/communication.py

1. `AgentNetwork`: a method exchanges things itself (here world-model latents) over a perfect, then a lossy channel
2. `CommunicationWrapper` in a World: each agent observes the messages that reached it; datasets record them
3. a topology that changes with the agents' positions (they only hear agents within range)
"""
import tempfile

import numpy as np
from comm_core.channel import AWGNChannel, IdentityChannel

import stable_marl as sm
from stable_marl.comm import AgentNetwork, communication, within_range

# 1. a method's own communication: 3 agents share 16-dim latents, perfect channel vs 30% packet loss
latents = {a: np.random.default_rng(a).normal(size=16).astype(np.float32) for a in range(3)}
for name, channel in (('perfect', IdentityChannel()), ('lossy', AWGNChannel(snr_db=5, packet_error_rate=0.3))):
    net = AgentNetwork(3, channel=channel, seed=0)
    for step in range(100):
        received = net.exchange(latents, type='latent')        # received[i][j]: agent j's latent, as agent i got it
    m = net.metrics
    print(f'{name:8s} latents: {m.successful}/{m.transmissions} delivered ({100 * m.packet_loss:.0f}% lost), '
          f'{m.bits_sent / 8 / 1024:.1f} KiB sent; agent 0 last heard from {sorted(received[0])}')
net = AgentNetwork(3, bits_per_element=4, seed=0)                    # e.g. latents quantized to 4 bits
net.exchange(latents)
print(f'quantized to 4 bits per number: {net.metrics.bits_sent} bits per round instead of {6 * 16 * 32}')

# 2. communication as part of the environment: FindGoal agents tell each other where they are
with tempfile.TemporaryDirectory() as tmp:
    for name, channel in (('perfect', IdentityChannel()), ('lossy', AWGNChannel(snr_db=5, packet_error_rate=0.3))):
        world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, agents=3, size=9, num_obstacles=0, max_episode_steps=30,
                         pre_wrappers=[communication(message_keys=('position', 'direction'), channel=channel)])
        world.set_policy(sm.RandomPolicy(seed=0))
        world.collect(f'{tmp}/{name}.h5', episodes=8, seed=0, progress=False)
        ds = sm.HDF5Dataset(path=f'{tmp}/{name}.h5', num_steps=1)
        mask = np.concatenate([ds[i]['message_mask'].numpy() for i in range(len(ds))])   # (steps, receiver, sender)
        heard = mask.sum(-1).mean() / 2                                                  # out of the 2 other agents
        print(f'{name:8s} world: messages {tuple(ds[0]["messages"].shape[1:])} per step (receiver, sender, D), '
              f'{100 * heard:.0f}% of the other agents heard per step')

# 3. agents only hear the agents within 3 cells (the topology follows their positions)
world = sm.World('MultiGrid-FindGoal-15x15-v0', num_envs=4, agents=3, size=15, num_obstacles=0, max_episode_steps=30,
                 pre_wrappers=[communication(message_keys=('position',), topology=within_range(3))])
world.set_policy(sm.RandomPolicy(seed=0))
heard = []
def on_step(world, mask):
    heard.append(world.infos['message_mask'][mask, 0].sum(-1).mean())
for _ in world._run_iter(8, seed=0, mode='auto', on_step=on_step):
    pass
print(f'range-limited (3 cells, 15 x 15 grid): {np.mean(heard):.2f} of 2 other agents heard per step on average')
