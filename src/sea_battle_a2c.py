import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from src.sea_battle import SeaBattleBoard, SeaBattleEnvironment
from lib.utils import state_to_one_hot_3
from torch.utils.tensorboard import SummaryWriter


class ActorCriticNetwork(nn.Module):
    def __init__(self, device, layer_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(layer_size[0], layer_size[1])
        self.fc2 = nn.Linear(layer_size[1], layer_size[2])

        self.actor = nn.Linear(layer_size[2], layer_size[3])
        self.critic = nn.Linear(layer_size[2], 1)

        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

    def get_action_probs(self, x):
        actor_output = self.actor(x)
        action_probs = torch.softmax(actor_output, dim=-1)
        return action_probs

    def get_state_value(self, x):
        state_value = self.critic(x)
        return state_value


class A2C:
    def __init__(self, architecture, hyperparameters, training, device, name):
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        self.training = training
        self.device = device
        self.name = name

        self.network = ActorCriticNetwork(device, architecture['layer_sizes']).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=hyperparameters['learning_rate'])

        self.writer = SummaryWriter(log_dir=f'./runs/{name}', flush_secs=1)
        self.global_step = 0

    def run(self, model_save_path):
        env = SeaBattleEnvironment()

        for episode in range(self.training['num_episodes']):
            state = env.reset()
            state = state_to_one_hot_3(state)
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                for _ in range(self.hyperparameters["num_steps"]):
                    if done:
                        break
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                    x = self.network(state_tensor.unsqueeze(0)).squeeze(0)
                    action_probs = self.network.get_action_probs(x)
                    
                    # Set the probability of invalid actions to zero
                    valid_actions = env.get_random_valid_action()
                    action_probs_np = action_probs.cpu().detach().numpy()
                    action_probs_np[~valid_actions] = 0
                    
                    # Normalize the remaining action probabilities
                    action_probs_np /= np.sum(action_probs_np)
                    action = np.random.choice(np.arange(action_probs_np.shape[0]), p=action_probs_np)

                    next_state, reward, done, _ = env.step(action)
                    next_state = state_to_one_hot_3(next_state)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                    next_x = self.network(next_state_tensor.unsqueeze(0)).squeeze(0)

                    state_value = self.network.get_state_value(x)
                    next_state_value = self.network.get_state_value(next_x)

                    advantage = reward + self.hyperparameters['gamma'] * (1 - done) * next_state_value - state_value

                    # Calculate entropy to encourage exploration
                    entropy = -torch.sum(action_probs * torch.log(action_probs))

                    actor_loss = -torch.log(action_probs[action]) * advantage.detach()
                    critic_loss = advantage.pow(2)

                    # Include the entropy term in the loss function
                    loss = actor_loss + critic_loss - self.hyperparameters["entropy_coefficient"] * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_reward += reward
                    episode_steps += 1
                    state = next_state

            self.writer.add_scalar('Episode Reward', episode_reward, episode)

            # Log the number of steps per episode
            self.writer.add_scalar('Steps', episode_steps, episode)
        torch.save(self.network.state_dict(), os.path.join(model_save_path, self.name))

    
    def close(self):
        self.writer.close()

