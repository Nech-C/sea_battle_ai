import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
from collections import deque
from src.sea_battle import SeaBattleBoard, SeaBattleEnvironment
from lib.utils import ReplayBuffer, Training_instance
from typing import Dict
from lib.utils import state_to_one_hot_3
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, device: torch.device, layer_size: list):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])
        self.fc3 = nn.Linear(layer_size[2], layer_size[3])
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)
        return x
    
class PolicyGradient(Training_instance):
    def __init__(self, config, device):
        # storing parameters 
        self.model_name = config["model_name"]
        self.architecture = config["architecture"]
        self.hyperparameters = config["hyperparameters"]
        self.reward_function = config["reward_function"]
        self.training = config["training"]
        self.device = device
        
        # define the policy netwrok
        self.policy_network = Policy(device, config["architecture"]["layer_sizes"])
        self.memory = []
        self.optimizer = optim.Adam(self.policy_network.parameters, lr=self.hyperparameters["training_rate"])
    
    def get_action_n_prob(self, state):
        state_tensor = torch.tensor(state_to_one_hot_3(state)).to(self.device)
        action_probs = self.policy_network(state_tensor)
        a = Categorical(action_probs)
        action = a.sample()
        
        return action.item(), a.log_prob(action)
    
    
    
    def update_policy(self, log_probs: list, rewards: list):
        # rename variable for better reabablility
        gamma = self.hyperparameters["gamma"]
        optimizer = self.optimizer
        
        # create local variables for policy update
        R = 0
        policy_loss = []
        returns = deque()
        
        # compute values for each step
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns, dtype=None)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        for log_prob, R in zip(log_probs, R):
            policy_loss.append(-log_prob * R)
        
        # backpropagation
        optimizer.zero_grad()
        policy_loss = 1/len(policy_loss) * torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # delete all elements in the two lists
        del log_probs[:]
        del rewards[:]
        
        
        
    def run(self):
        env: SeaBattleEnvironment = SeaBattleEnvironment(rewards=self.reward_function)
        
        for episode in range(self.training["num_episodes"]):
            state = env.reset()
            ep_reward = 0
            done = False
            log_probs = []
            rewards = []
            step_count = 0
            for _ in range(self.hyperparameters["max_step"]):
                step_count += 1
                
                action, log_prob = self.get_action_n_prob(state)
                state, reward, done, _, _ = env.step(action)
                
                rewards.append(reward)
                log_probs.append(log_prob)
                ep_reward += reward
                
                if done:
                    break
            
            self.update_policy(log_probs, rewards)
            
            # print to the stdout
            print(f"episode: {episode}\t episode reward: {ep_reward}\t episode step count: {step_count}")
            
                
        
                