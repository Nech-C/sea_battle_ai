import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
from collections import deque
from sea_battle import SeaBattleBoard, SeaBattleEnvironment

# Constants
BUFFER_SIZE = 15000
NUM_SQUARES = 49
NUM_EPISODES = 2500
BATCH_SIZE = 1024
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.00007
MAX_INVALID_ACTION_COUNT = 10
HIDDEN_LAYER_SIZE = [512, 256]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """A simple replay buffer implementation to store and sample experiences."""

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DeepQNetwork(nn.Module):
    def __init__(self, num_squares, device, hidden_layer_size=HIDDEN_LAYER_SIZE):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_squares * 5, hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.fc3 = nn.Linear(hidden_layer_size[1], num_squares)
        self.to(device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def state_to_one_hot(state, last_action=None):
    num_squares = state.size
    one_hot_state = np.zeros((num_squares, 5), dtype=np.float32)

    for i, square in enumerate(state):
        if square == -1:
            one_hot_state[i, 0] = 1
        elif square == 0:
            one_hot_state[i, 1] = 1
        elif square == 1:
            one_hot_state[i, 2] = 1
        elif square == 2:
            one_hot_state[i, 3] = 1

        if last_action == i and square != -1:
            one_hot_state[i, 4] = 1
        else:
            one_hot_state[i, 4] = 0

    return one_hot_state.flatten()

def choose_action(state, dqn, device, epsilon, env, invalid_action_count):
    """Choose an action based on the current state, epsilon value and DQN."""

    if np.random.rand() < epsilon or invalid_action_count >= MAX_INVALID_ACTION_COUNT:
        action = env.action_space.sample()
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = dqn(state_tensor.unsqueeze(0))
        action = torch.argmax(q_values).item()
    return action

def update_dqn(dqn, memory, optimizer, device, batch_size=BATCH_SIZE, gamma=GAMMA):
    """Update the DQN based on a batch of experiences from the memory buffer."""

    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    current_q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = dqn(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones.float())
    loss = nn.MSELoss()(current_q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_dqn(dqn, device, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, learning_rate=LEARNING_RATE):
    """Train the DQN with the given hyperparameters."""

    env = SeaBattleEnvironment()
    memory = ReplayBuffer(BUFFER_SIZE)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state = env.reset().flatten()
        state = state_to_one_hot(state)
        done = False
        steps_to_complete = 0
        invalid_action_count = 0
        loss = None

        while not done:
            action = choose_action(state, dqn, device, epsilon, env, invalid_action_count)
            next_state, reward, done, _ = env.step(action=action)
            next_state = state_to_one_hot(next_state.flatten(), action)

            if np.array_equal(state, next_state):
                invalid_action_count += 1
            else:
                invalid_action_count = 0

            memory.push(state, action, reward, next_state, done)
            state = next_state
            steps_to_complete += 1

            if len(memory) > batch_size:
                loss = update_dqn(dqn, memory, optimizer, device)

        if episode % 20 == 0 and loss is not None:
            print(f"Episode: {episode}, Loss: {loss}, Epsilon: {epsilon}, Steps to complete: {steps_to_complete}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

def save_dqn(dqn, model_path="trained_model.pth"):
    """Save the DQN model to the specified path."""

    torch.save(dqn.state_dict(), model_path)

def test(dqn, device, num_test_episodes=100, max_steps=30, random_play=False):
    env = SeaBattleEnvironment()
    rewards = []
    steps = []

    for episode in range(num_test_episodes):
        state = env.reset()
        state = state.flatten()
        state = state_to_one_hot(state)
        episode_reward = 0
        done = False
        episode_steps = 0

        while not done and episode_steps < max_steps:
            if random_play:
                # Choose a random valid action
                action = env.get_random_valid_action()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                with torch.no_grad():
                    action_tensor = dqn(state_tensor.unsqueeze(0)).argmax(dim=1)

                action = int(action_tensor.item())
                
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()
            next_state = state_to_one_hot(next_state)

            episode_reward += reward
            state = next_state
            episode_steps += 1

        rewards.append(episode_reward)
        steps.append(episode_steps)

    average_reward = sum(rewards) / len(rewards)
    average_steps = sum(steps) / len(steps)
    print(f"Average reward over {num_test_episodes} episodes: {average_reward:.2f}")
    print(f"Average steps to finish the game over {num_test_episodes} episodes: {average_steps:.2f}")
    return average_reward, average_steps

def main():
    # Create an argument parser
    import argparse
# ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sea Battle DQN")
    parser.add_argument("command", choices=["train", "test"], help="either 'train' or 'test'")
    parser.add_argument("episodes", type=int, help="number of episodes to run")
    parser.add_argument("model_path", help="path to the model file")
    parser.add_argument("--random", action="store_true", help="use random actions for testing")
    args = parser.parse_args()



    # Create the DQN model
    dqn = DeepQNetwork(NUM_SQUARES, device)

    if args.action == "train":
        train_dqn(dqn, device, num_episodes=args.episodes)
        save_dqn(dqn, model_path=args.model_path)

    elif args.action == "test":
        dqn.load_state_dict(torch.load(args.model_path))
        dqn.to(device)
        dqn.eval()
        average_reward, average_steps = test(dqn, device, num_test_episodes=args.episodes, random_play=args.random)
        print(f"Average reward: {average_reward}, Average steps: {average_steps}")

if __name__ == "__main__":
    main()

