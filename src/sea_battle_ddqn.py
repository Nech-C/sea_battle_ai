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




class DeepQNetwork1(nn.Module,):
    def __init__(self, device, layer_size):
        super(DeepQNetwork1, self).__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])
        self.fc3 = nn.Linear(layer_size[2], layer_size[3])
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DoubleDQN():
    """_summary_
    """
    def __init__(self, architecture: Dict, hyperparameters: Dict, training: Dict, device, name):
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.training = training
        self.device = device
        self.q_network = DeepQNetwork1(device, architecture["layer_sizes"]).to(device)
        self.target_network = DeepQNetwork1(device, architecture["layer_sizes"]).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hyperparameters["learning_rate"])
        self.memory = ReplayBuffer(hyperparameters["memory_capacity"])
        self.update_counter = 0
        self.epsilon = hyperparameters["epsilon_start"]
        self.loss_fn = nn.MSELoss()
        self.gamma = hyperparameters["gamma"]
        self.name = name
        self.epsilon_decay = hyperparameters["epsilon_decay"]
    
    def run(self, model_save_path: str):
        """ it runs for the number of episodes specified in the training dict
        """
        hyperparameters = self.hyperparameters
        architecture = self.architecture
        training = self.training
        memory = self.memory
        env = SeaBattleEnvironment()
        
        tot_steps = 0
        for episode in range(training["num_episodes"]):
            state = env.reset()
            state = state_to_one_hot_3(state)
            episode_reward = 0
            done = False
            episode_steps = 0
            invalid_action_count = 0
            training_loss = None
            
            while not done:
                action = self.get_action(state, env, invalid_action_count)  # Pass the invalid_action_count
                next_state, reward, done, _= env.step(action)
                next_state = state_to_one_hot_3(next_state)
                memory.push((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if not done and reward == -1:
                        invalid_action_count += 1
                else:
                    invalid_action_count = 0
            
            # update the epsilon value
            self.update_epsilon()
            tot_steps += episode_steps
            # train the network
            if len(memory) > hyperparameters["batch_size"]:
                training_loss = self.update_dqn()
                
            # display training information
            if episode % 100 == 0:
                training_loss_str = "{:.5f}".format(training_loss) if training_loss is not None else "N/A"
                print(f"episode: {episode} | episode reward: {episode_reward:.2f} | training loss: {training_loss_str} | ave steps: {tot_steps/100} | epsilon: {self.epsilon:.2f}")
                tot_steps = 0
                
        torch.save(self.q_network.state_dict(), os.path.join(model_save_path, self.name))
    
    def get_action(self, state, env, invalid_action_count):
        """ it returns an action"""
        if np.random.rand() < self.epsilon or invalid_action_count >= self.training['invalid_action_limit']:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.q_network(state_tensor.unsqueeze(0))
            action = torch.argmax(q_values).item()
        return action
    
    def update_target_network(self):
        """ it updates the target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.hyperparameters["tau"] * local_param.data + (1.0 - self.hyperparameters["tau"]) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.hyperparameters["epsilon_end"], self.epsilon * self.epsilon_decay)
    
    def update_dqn(self):
        """ it updates the q_network and target_network"""
        memory = self.memory
        hyperparameters = self.hyperparameters
        device = self.device
        batch = memory.sample(hyperparameters["batch_size"])
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
                
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        next_q_values = self.q_network(next_states).detach()
        _, best_actions = next_q_values.max(dim=1, keepdim=True)
        
        target_next_q_values = self.target_network(next_states).detach()
        target_q_values = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * target_next_q_values.gather(1, best_actions)
        
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % hyperparameters["target_update_freq"] == 0:
            self.update_target_network()
        
        return loss.item()

        
