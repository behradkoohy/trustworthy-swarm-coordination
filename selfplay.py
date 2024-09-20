import numpy as np
import torch
from torch import nn

from CRL2_Swarm import Agent, CategoricalMasked

from WorldEnvOHE import WorldEnv


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.actor = nn.Sequential(
            self._layer_init(nn.Conv2d(5, 16, 4, padding=1)),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(16, 16, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(1296, 128)),
            nn.ReLU(),
            self._layer_init(nn.Linear(128, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            self._layer_init(nn.Conv2d(5, 16, 4, padding=1)),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(16, 16, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(1296, 128)),
            nn.ReLU(),
            self._layer_init(nn.Linear(128, 1)),
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, action_masks=None):
        logits = self.actor(x)
        probs = CategoricalMasked(logits=logits, masks=action_masks)
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            (self.critic(x)),
        )



env = WorldEnv()
obs, info = env.reset()
env.show_grid(env.all_grids)
# for step in range(0, 300):
#     actions = {
#         agent: int(input(f"{agent=}")) for agent in env.agents
#     }
#     observations, rewards, terms, truncs, infos = env.step(actions)
#     env.show_grid(env.all_grids)
#     print(f'{observations["drone_0"].shape=}')
#     print([env.get_action_masks(ob, n) for n, ob in enumerate(observations.values())])
#     print(rewards)
# env.close()

grids = env.all_grids
print(f'{len(grids)=}')
print(f'{np.sum([grids[3], grids[4]], axis=0).shape=}')

print(f'{np.array(env.all_grids)=}')
print(f'{env.get_observe("drone_0").dtype}')

agent = Agent(4)
agent.actor.load_state_dict(torch.load('actor.model', weights_only=True))
agent.critic.load_state_dict(torch.load('critic.model', weights_only=True))

breakpoint()

