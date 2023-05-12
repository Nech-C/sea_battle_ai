import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
from src.sea_battle2 import SeaBattleEnv, Observation, CELL_MAPPING, GuessResult
from lib.utils import Training_instance
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from lib.utils import load_reward_function

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
        self.reward_function = load_reward_function(config["reward_function"])
        self.device = device
        
        # define the policy netwrok
        self.policy_network = Policy(device, config["architecture"]["layer_sizes"])
        self.memory = []
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.hyperparameters["learning_rate"])
    
        self.writer = SummaryWriter(log_dir=f'./runs/{self.model_name}', flush_secs=1)
    def get_action_n_prob(self, state: np.ndarray):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.policy_network(state_tensor)
        a = Categorical(action_probs)
        action = a.sample()
        
        return action.item(), a.log_prob(action)
    
    def calculate_base_line(self, state: np.ndarray, ships):
        """ this function is used to get the baseline for a given state
        
        Args:
            state (np.ndarray): a numpy array of the state(not one-hot encoded)
        
        Returns:
            float: the baseline for the given state
        
        
        """
        empty = np.count_nonzero(state == CELL_MAPPING['empty']) + 0.0000000001
        return self.reward_function[GuessResult.SHIP_HIT] * (ships / (empty + ships)) + self.reward_function[GuessResult.NO_HIT] * (empty / (empty + ships))

       
    def update_policy(self, log_probs: list, rewards: list, base_lines: list):
        # rename variable for better reabablility
        gamma = self.hyperparameters["gamma"]
        optimizer = self.optimizer
        
        # create local variables for policy update
        R = 0
        policy_loss = []
        returns = deque()
        
        # compute values for each step
        rewards = [r - b for r, b in zip(rewards, base_lines)]
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.appendleft(R)
        returns = torch.tensor(returns, dtype=None)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(log_prob * R)
        
        # backpropagation
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).mean()
        policy_loss.backward()
        optimizer.step()
        
        # delete all elements in the two lists
        del log_probs[:]
        del rewards[:]
        
        
        
    def run(self):
        env: SeaBattleEnv = SeaBattleEnv(reward_func=self.reward_function)
        
        for episode in range(self.training["num_episodes"]):
            state = env.reset()
            ep_reward = 0
            done = False
            log_probs = []
            rewards = []
            base_lines = []
            step_count = 0
            for _ in range(self.hyperparameters["max_step"]):
                step_count += 1
                
                action, log_prob = self.get_action_n_prob(state.get_one_hot_encoding().flatten())
                state, reward, done, info = env.step(action)
                base_line = self.calculate_base_line(state.get_board(), info['ships'])
                base_lines.append(base_line)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                ep_reward += reward
                
                if done:
                    break
            
            self.update_policy(log_probs, rewards, base_lines)
            
            # print to the stdout
            print(f"episode: {episode}\t episode reward: {ep_reward}\t episode step count: {step_count}")
            _, tot_guesses, effective_guesses = env.get_stats()
            self.writer.add_scalar('episode_reward', ep_reward, episode)
            self.writer.add_scalar('episode_step_count', step_count, episode)
            self.writer.add_scalar('total_guesses', tot_guesses, episode)
            self.writer.add_scalar('effective_guesses', effective_guesses, episode)

    def close(self, model_save_path):
        torch.save(self.policy_network.state_dict(), os.path.join(model_save_path, self.model_name))
        self.writer.close()
        