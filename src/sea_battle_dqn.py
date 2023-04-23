from sea_battle import SeaBattleBoard,  SeaBattleEnvironment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random



# Create ReplayBuffer class
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
# Define the DeepQNetwork class
class DeepQNetwork(nn.Module):
    def __init__(self, num_squares, device, hidden_layer_size=[512, 256]):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_squares, hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.fc3 = nn.Linear(hidden_layer_size[1], num_squares)
        self.to(device)

        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

# Define the training function
def train(dqn, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, device):
    env = SeaBattleEnvironment()
    memory = ReplayBuffer(10000)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    num_episodes = 2500
    
    MAX_INVALID_ACTION_COUNT = 10
    invalid_action_count = 0
    loss = None
    for episode in range(num_episodes):
        # initialize the environment
        state = env.reset()
        state = state.flatten()
        done = False
        steps_to_complete = 0 # the number of steps taken to complete the episode
        
        while not done:
            if np.random.rand() < epsilon or invalid_action_count >= MAX_INVALID_ACTION_COUNT:
                action = env.action_space.sample()
            else:
                q_values = dqn(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()
            
            next_state, reward, done, _ = env.step(action=action)
            next_state = next_state.flatten()
            
            if np.array_equal(state, next_state):
                invalid_action_count += 1
            else:
                invalid_action_count = 0

            #  push the transition to the memory buffer
            memory.push(state, action, reward, next_state, done)
            
            # update the state and increment the step counter
            state = next_state
            steps_to_complete += 1
            
            # train the DQN on the memory buffer
            if len(memory) > batch_size:
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
                
        if episode % 20 == 0 and loss is not None:
                    print(f"Episode: {episode}, Loss: {loss.item()}, Epsilon: {epsilon}, Steps to complete: {steps_to_complete}")        
                

                
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay      
            
def main():
    # Define hyperparameters
    batch_size = 2048
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.00007
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the DeepQNetwork class
    num_squares = 49  # Assuming a 10x10 grid
    dqn = DeepQNetwork(num_squares, device)

    # Train the DQN
    train(dqn, batch_size=batch_size, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, device=device)

    torch.save(dqn.state_dict(), "trained_model.pth")

    # Episode: 100, Loss: 2173297664.0, Epsilon: 0.6057704364907278
    # Episode: 200, Loss: 57742004.0, Epsilon: 0.3669578217261671
    # Episode: 300, Loss: 5281507.5, Epsilon: 0.22229219984074702
    
def test(dqn, device, num_test_episodes=100, max_steps=30):
    env = SeaBattleEnvironment()
    rewards = []
    steps = []

    for episode in range(num_test_episodes):
        state = env.reset()
        state = state.flatten()
        episode_reward = 0
        done = False
        episode_steps = 0

        while not done and episode_steps < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                action_tensor = dqn(state_tensor.unsqueeze(0)).argmax(dim=1)

            action = int(action_tensor.item())
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten()

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



def test_model(model_path, device):
    # Instantiate the DeepQNetwork class
    num_squares = 49  # Assuming a 10x10 grid
    dqn = DeepQNetwork(num_squares, device,[128, 128])

    # Load the trained model
    dqn.load_state_dict(torch.load(model_path))
    dqn.to(device)
    dqn.eval()
    
    # Test the model
    average_reward = test(dqn, device)
    print(f"Average reward: {average_reward}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "trained_model.pth"

    # Uncomment the following line if you want to train the model again
    main()

    # test_model(model_path, device)
