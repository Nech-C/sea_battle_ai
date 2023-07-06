import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from src.sea_battle import SeaBattleEnv, Observation, GuessResult
from lib.utils import Training_instance, ReplayBuffer, load_reward_function
from torch.utils.tensorboard import SummaryWriter
import numpy.random as npr
from torch.distributions import Categorical


class PolicyShared(nn.Module):
    def __init__(self, shared_layers, actor_layers, critic_layers):
        super(PolicyShared, self).__init__()
        # shared layers
        self.fc_shared = nn.Linear(shared_layers[0], shared_layers[1])
        nn.init.kaiming_normal_(self.fc_shared.weight, nonlinearity='relu')

        # actor's layers
        self.fc1_actor = nn.Linear(shared_layers[1], actor_layers[0])
        nn.init.kaiming_normal_(self.fc1_actor.weight, nonlinearity='relu')
        self.fc2_actor = nn.Linear(actor_layers[0], actor_layers[1])
        nn.init.kaiming_normal_(self.fc2_actor.weight, nonlinearity='relu')

        # critic's layers
        self.fc1_critic = nn.Linear(shared_layers[1], critic_layers[0])
        nn.init.kaiming_normal_(self.fc1_critic.weight, nonlinearity='relu')
        self.fc2_critic = nn.Linear(critic_layers[0], critic_layers[1])
        nn.init.kaiming_normal_(self.fc2_critic.weight, nonlinearity='relu')

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        # shared layer
        x = F.relu(self.fc_shared(x))

        # actor: 
        action_prob = self.fc1_actor(x)
        action_prob = F.softmax(self.fc2_actor(action_prob), dim=-1)

        # critic:
        value = F.relu(self.fc1_critic(x))
        value = self.fc2_critic(value)

        if self.training:
            return action_prob, value
        else:
            return action_prob


class A2C(Training_instance):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.network = self.init_policy().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hyperparameters["learning_rate"])

    def init_policy(self) -> nn.Module:
        if self.architecture['nn_type'] == 'shared':
            # extract network info
            input_size = self.architecture['board_width'] * self.architecture['board_length']
            shared_hidden_layers = self.architecture['shared_hidden']
            shared = [input_size * 3, *shared_hidden_layers]

            actor = self.architecture['actor']['actor_hidden_layers'] + [input_size]

            critic = self.architecture['critic']['critic_hidden_layers'] + [1]

            return PolicyShared(shared, actor, critic)

    def get_action(self, observation: Observation, invalid_count: int):
        if invalid_count >= self.hyperparameters['invalid_action_limit']:
            return observation.get_random_valid_action()

        state = observation.get_one_hot_encoding().flatten()
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        probs, value = self.network(state)
        m = Categorical(probs)
        action = m.sample()
        self.network.saved_actions.append((m.log_prob(action), value))
        return action.item()

    def update_network(self) -> float:
        R = 0
        saved_actions = self.network.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        gamma = self.hyperparameters['gamma']

        for r in self.network.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic loss
            value_losses.append(F.smooth_l1_loss(value.squeeze(1), torch.tensor([R]).to(self.device)))

        self.optimizer.zero_grad()

        loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

        self.optimizer.step()

        del self.network.rewards[:]
        del self.network.saved_actions[:]

        return loss.item()

    def run(self):
        # step up the environment
        shape = (self.architecture['board_length'], self.architecture['board_width'])
        env = SeaBattleEnv(shape, self.reward_function)

        for episode in range(self.training['num_episodes']):
            invalid_action_count = 0
            obs = env.reset()
            done = False
            while not done:
                action = self.get_action(obs, invalid_action_count)
                if invalid_action_count >= self.hyperparameters['invalid_action_limit']:
                    invalid_action_count = 0

                obs, reward, done, info = env.step(action)
                self.network.rewards.append(reward)

                if reward == self.reward_function[GuessResult.INVALID_GUESS]:
                    invalid_action_count += 1

            loss = self.update_network()
            if episode % 20 == 0:
                self.update_tensorboard(episode, env, loss, eps=None)
