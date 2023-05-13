import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque
from src.sea_battle import SeaBattleEnv, Observation, CELL_MAPPING, GuessResult
from lib.utils import Training_instance, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from lib.utils import load_reward_function

class DQN(nn.Module):
    def __init__(self, device: torch.device, layer_size: list):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])
        self.fc3 = nn.Linear(layer_size[2], layer_size[3])
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDQNAgent(Training_instance):
    def __init__(self, config, device):
        # Store parameters
        self.model_name = config["model"]["model_name"]
        self.architecture = config["architecture"]
        self.hyperparameters = config["hyperparameters"]
        self.reward_function = config["reward_function"]
        self.training = config["training"]
        self.reward_function = load_reward_function(config["reward_function"])
        self.device = device

        # Define the DQN networks
        network_arch = [config['architecture']['input_size'], *config['architecture']['hidden_sizes'], config['architecture']['output_size']]
        self.online_network = DQN(device, network_arch)
        self.target_network = DQN(device, network_arch)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.memory = ReplayBuffer(self.hyperparameters["buffer_size"])
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.hyperparameters["learning_rate"])

        self.writer = SummaryWriter(log_dir=f'./runs/{self.model_name}', flush_secs=1)

    def get_action(self, state: np.ndarray, eps):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        online_q_values = self.online_network(state_tensor)

        if np.random.rand() < eps:
            return np.random.choice(online_q_values.shape[-1])
        else:
            return torch.argmax(online_q_values).item()

    def update_network(self):
        if len(self.memory) < self.hyperparameters["batch_size"]:
            return

        batch = self.memory.sample(self.hyperparameters["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device).unsqueeze(-1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        online_q_values = self.online_network(states).gather(1, actions)

        with torch.no_grad():
            next_online_q_values = self.online_network(next_states)
            next_actions = torch.argmax(next_online_q_values, dim=-1).unsqueeze(-1)
            target_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.hyperparameters["gamma"] * target_q_values


        loss = nn.MSELoss()(online_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        eps_start = self.hyperparameters["epsilon_start"]
        eps_min = self.hyperparameters["epsilon_min"]
        eps_decay = self.hyperparameters["epsilon_decay"]

        env: SeaBattleEnv = SeaBattleEnv(reward_func=self.reward_function)

        for episode in range(self.training["num_episodes"]):
            state = env.reset().get_one_hot_encoding().flatten()
            ep_reward = 0
            done = False
            step_count = 0

            eps = eps_start * (eps_min / eps_start) ** (episode / eps_decay)

            for _ in range(self.hyperparameters['max_step']):
                if done:
                    break
                
                step_count += 1

                action = self.get_action(state, eps)
                next_state, reward, done, info = env.step(action)
                next_state = next_state.get_one_hot_encoding().flatten()

                self.memory.push((state, action, reward, next_state, done))
                self.update_network()

                state = next_state
                ep_reward += reward

            if episode % self.hyperparameters["target_update_frequency"] == 0:
                self.target_network.load_state_dict(self.online_network.state_dict())

            print(f"episode: {episode}\t episode reward: {ep_reward}\t episode step count: {step_count}")
            _, tot_guesses, effective_guesses = env.get_stats()
            self.writer.add_scalar('episode_reward', ep_reward, episode)
            self.writer.add_scalar('episode_step_count', step_count, episode)
            self.writer.add_scalar('total_guesses', tot_guesses, episode)
            self.writer.add_scalar('effective_guesses', effective_guesses, episode)

    def close(self, model_save_path):
        torch.save(self.online_network.state_dict(), os.path.join(model_save_path, self.model_name))
        self.writer.close()
