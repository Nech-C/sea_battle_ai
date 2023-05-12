import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
from collections import deque
from src.sea1.sea_battle import SeaBattleBoard, SeaBattleEnvironment
from lib.utils import ReplayBuffer, Training_instance
from typing import Dict
from lib.utils import state_to_one_hot_3
from torch.utils.tensorboard import SummaryWriter

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, device, layer_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])
        self.fc3 = nn.Linear(layer_size[2], layer_size[3])
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

class PolicyGradient():
    """Policy Gradient implementation for the Sea Battle game."""
    def __init__(self, architecture: Dict, hyperparameters: Dict, training: Dict, device, name, reward_fun):
        # Save arguments as instance variables
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.training = training
        self.device = device
        self.name = name
        self.reward_fun = reward_fun
        
        # Create the policy network
        self.policy_network = PolicyNetwork(device, architecture["layer_sizes"]).to(device)
        
        # Define variables and objects for training
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=hyperparameters["learning_rate"])
        self.memory = []
        
        # Tensorboard writer and global_step
        self.writer = SummaryWriter(log_dir=f'./runs/{name}', flush_secs=1)
        self.global_step = 0
    
    def run(self, model_save_path: str):
        """Runs the agent for the number of episodes specified in the training dict."""
        hyperparameters = self.hyperparameters
        training = self.training
        env = SeaBattleEnvironment(rewards=self.reward_fun)
        
        for episode in range(training["num_episodes"]):
            state = env.reset()
            state = state_to_one_hot_3(state)
            done = False
            episode_reward = 0
            episode_steps = 0
            
            counter = 0
            while not done:
                action_probs = self.get_action_probs(state)
                action = np.random.choice(np.arange(action_probs.shape[0]), p=action_probs.detach().cpu().numpy())
                next_state, reward, done, _= env.step(action)
                next_state = state_to_one_hot_3(next_state)

                self.memory.append((state, action, reward))
                episode_reward += reward
                state = next_state
                episode_steps += 1
                counter += 1
                if counter > hyperparameters["max_step"]:
                    break
                

            # Update the policy network
            self.update_policy()
            
            # Log the episode reward
            self.writer.add_scalar('Episode Reward', episode_reward, episode)

            # Log the number of steps per episode
            self.writer.add_scalar('Steps', episode_steps, episode)
            print(f"Episode {episode}: {episode_reward} steps: {episode_steps}")

        torch.save(self.policy_network.state_dict(), os.path.join(model_save_path, self.name))

    def close(self):
        self.writer.close()
    
    def get_action_probs(self, state):
        """Returns the action probabilities for a given state."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.policy_network(state_tensor.unsqueeze(0)).squeeze(0)
        return action_probs
    
    def base_line(state):
        """Returns the baseline for a given state."""
        ...
        
    
    def update_policy(self):
        """Updates the policy network using the collected experiences."""
        R = 0
        policy_loss = []
        returns = []
        # Compute the returns for each time step
        for t in reversed(range(len(self.memory))):
            state, action, reward = self.memory[t]
            R = self.hyperparameters["gamma"] * R + reward
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Standardize the returns

        for (state, action, reward), R in zip(self.memory, returns):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_probs = self.policy_network(state_tensor.unsqueeze(0)).squeeze(0)

            # Calculate the policy loss
            loss = -torch.log(action_probs[action]) * R
            policy_loss.append(loss)

        # Update the policy network
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()


        # Clear the memory
        self.memory = []


