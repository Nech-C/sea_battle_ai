import torch
import torch.nn as nn
import numpy as np
import os
from sea_battle import SeaBattleEnv, Observation
from lib.utils import Training_instance, ReplayBuffer, load_reward_function
from torch.utils.tensorboard import SummaryWriter
import numpy.random as npr
class DQN(nn.Module):
    def __init__(self, device, layer_size):
        super().__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])
        self.fc3 = nn.Linear(layer_size[2], layer_size[3])
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(Training_instance):
    def __init__(self, config: dict, device):
        super().__init__(config, device)
        network_size = [self.architecture['input_size'], *self.architecture['hidden_size'], self.architecture['output_size']]
        self.network = DQN(device, network_size)
        self.memory = ReplayBuffer(self.hyperparameters['buffer_size'])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hyperparameters['learning_rate'])
        
    def update_network(self):
        if len(self.memory) < self.hyperparameters['batch_size']:
            return None
        batch = self.memory.sample(self.hyperparameters['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        
        current_q_values = self.network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.network(next_states).max(1)[0]
        target_q_values = rewards + self.hyperparameters['gamma'] * next_q_values * (1 - dones)
        target_q_values = target_q_values.unsqueeze(1)
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def get_action(self, observation: Observation, invalid_count: int, eps: float):
        if invalid_count >= self.hyperparameters['invalid_action_limit']:
            return observation.get_random_valid_action()
        if npr.rand() < eps:
            return observation.get_random_valid_action()
        else:
            state = observation.get_one_hot_encoding().flatten()
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return torch.argmax(self.network(state)).item()
    
    def update_tensor_board(self, episode: int, env: SeaBattleEnv, loss, eps):
        step_count, tot_guesses, effective_guesses, = env.get_stats()
        if loss is not None:
            self.writer.add_scalar('Loss/train', loss, episode)
        self.writer.add_scalar('Steps/train', step_count, episode)
        self.writer.add_scalar('Guesses/train', tot_guesses, episode) 
        self.writer.add_scalar('Effective guesses/train', effective_guesses, episode)
        self.writer.add_scalar('Epsilon/train', eps, episode)
    def run(self):
        # set up environment
        shape = (self.architecture['board_length'], self.architecture['board_width'])
        env = SeaBattleEnv(shape, self.reward_function)
        eps = self.hyperparameters['eps_start']
        eps_min = self.hyperparameters['eps_min']
        eps_decay = self.hyperparameters['eps_decay']
        
        for episode in range(self.training['num_episodes']):
            invalid_action_count = 0
            obs = env.reset()
            done = False
            eps = eps * eps_decay
            
            while not done:
                action = self.get_action(obs, invalid_action_count, max(eps_min, eps))
                next_obs, reward, done, _ = env.step(action)
                state = obs.get_one_hot_encoding().flatten()
                next_state = next_obs.get_one_hot_encoding().flatten()
                self.memory.push((state, action, reward, next_state, done))
                obs = next_obs
            
            loss =self.update_network()
            self.update_tensor_board(episode, env, loss, eps)
    def close(self, save_path: str):
        super().close(save_path)