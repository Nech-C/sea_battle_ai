# sea_ppo.py: 
import torch
import torch.nn as nn
import numpy as np
from src.sea_battle import SeaBattleEnv, Observation, GuessResult
from lib.utils import Training_instance, ReplayBuffer, load_reward_function
from torch.distributions import Categorical
import torch.nn.utils as utils

class PolicyNetwork(nn.Module):
    def __init__(self, layer_dimensions):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dimensions) - 1):
            layer = nn.Linear(layer_dimensions[i], layer_dimensions[i+1])
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)
        
    def forward(self, x):
        for layer in self.layers[:-1]:  # excluding the last layer
            x = torch.nn.functional.relu(layer(x))  # apply ReLU activation using the functional API
        x = torch.nn.functional.softmax(self.layers[-1](x), dim=-1)  # apply Softmax to the output of the last layer
        return x


class AdvantageNetwork(nn.Module):
    def __init__(self, layer_dimensions):
        super(AdvantageNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layer_dimensions) - 1):  # Subtract 1 to avoid going out of index
            layer = nn.Linear(layer_dimensions[i], layer_dimensions[i+1])
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:  # excluding the last layer
            x = torch.nn.functional.relu(layer(x))  # apply ReLU activation using the functional API
        x = self.layers[-1](x)  # No activation for the last layer
        return x


class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []


    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()


class PPO(Training_instance):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.actor, self.critic = self.init_policy()
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.hyperparameters['learning_rate']},
            {'params': self.critic.parameters(), 'lr': self.hyperparameters['learning_rate']}
        ])
        self.clip_epsilon = self.hyperparameters['clip_epsilon']
        self.buffer = PPOBuffer()  # Assuming ReplayBuffer implementation is similar to RolloutBuffer in reference

    def init_policy(self) -> nn.Module:
        input_dimension = self.architecture['board_width'] * self.architecture['board_length']
        policy_dimensions = [input_dimension] + self.architecture['actor']['actor_hidden_layers'] + [input_dimension]
        advantage_dimensions = [input_dimension] + self.architecture['critic']['critic_hidden_layers'] + [1]
        return PolicyNetwork(policy_dimensions), AdvantageNetwork(advantage_dimensions)
    
    def select_action(self, observation: Observation, invalid_count) -> int:
        state = torch.tensor(observation.board, dtype=torch.float32).view(-1).to(self.device)
        action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        if invalid_count >= self.hyperparameters['invalid_action_limit']:
            action = torch.tensor(observation.get_random_valid_action()).to(self.device)     
        else:
            action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_log_prob)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.hyperparameters["gamma"] * discounted_reward)
            rewards.insert(0, discounted_reward)
        moving_loss = 0
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.stack(self.buffer.states).to(self.device)
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.hyperparameters['K_epochs']):

            # Evaluating old actions and values
            action_probs = self.actor(old_states)
            state_values = self.critic(old_states)
            action_distribution = Categorical(action_probs)
            action_logprobs = action_distribution.log_prob(old_actions)
            dist_entropy = action_distribution.entropy()

            # torch.nninding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            # calculate advantages
            advantages = rewards - state_values.detach()

            # torch.nninding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2)
            
            # final loss of clipped objective PPO
            mse_loss = nn.MSELoss()
            critic_loss = mse_loss(state_values.squeeze(), rewards)
            
            entropy_bonus = 0.01 * dist_entropy

            loss = actor_loss + 0.5 * critic_loss - entropy_bonus

            # take gradient step
            self.optimizer.zero_grad()
            moving_loss = moving_loss * 0.8 + loss.mean().item() * 0.2
            loss.mean().backward()
            utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()
        return moving_loss
        
    def run(self):
        shape = (self.architecture['board_length'], self.architecture['board_width'])
        env = SeaBattleEnv(shape, self.reward_function)
        
        for episode in range(self.training['num_episodes']):
            obs = env.reset()
            done = False
            invalid_action_count = 0

            while not done:
                action = self.select_action(obs, invalid_action_count)  # Updated method name to select_action
                if invalid_action_count >= self.hyperparameters['invalid_action_limit']:
                    invalid_action_count = 0
                
                new_obs, reward, done, info = env.step(action)
                
                # save data in buffer
                self.buffer.rewards.append(reward)

                
                if reward == self.reward_function[GuessResult.INVALID_GUESS]:
                    invalid_action_count += 1
                
                obs = new_obs

            # Update the policy
            loss = self.update()
            if episode % 20 == 0:
                self.update_tensorboard(episode, env, loss, eps=None)
                print(env.get_stats())

