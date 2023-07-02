# Sea Battle AI - Readme

This repository contains AI agents that can play the Sea Battle game from GamePigeon. Currently, the following reinforcement learning algorithms have been implemented:

1. DQN (Deep Q-Network)
2. A2C (Advantage Actor-Critic)
3. VPG (Vanilla Policy Gradient)
4. PPO (Proximal Policy Optimization)

## Setup

To run the scripts, you need to have Python installed on your system. The scripts are compatible with Python 3.6 or later.

1. Clone the repository:

   ```shell
   git clone https://github.com/Nech-C/sea_battle_ai.git
   ```

2. Navigate to the cloned directory:

   ```shell
   cd sea_battle_ai
   ```

3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Training

To train the AI agents, you need to run the `trainer.py` script. Make sure to specify the configuration file using the `--config` flag.

```shell
python bin/trainer.py --config config/config-file.toml
```

Replace `config-file.toml` with the path to your desired configuration file.

## Evaluation

To evaluate the trained AI agents, you can run the `evaluator.py` script. Specify the model path using the `--model_path` flag and the configuration file using the `--config` flag.

```shell
python bin/evaluator.py --model_path models/sea2_dqn_v0.2.0.pth --config config/dqn_v0.2.0.toml
```

Replace `models/sea2_dqn_v0.2.0.pth` with the path to your trained model file, and `config/dqn_v0.2.0.toml` with the path to your configuration file.

## Configuration File

The configuration file (`config-file.toml`) contains various settings for training and evaluation, including hyperparameters, network architectures, and paths to save checkpoints and logs. You can modify these settings based on your requirements.

Here is an example of a configuration file (`config-file.toml`):

```toml
[model]
model_name = "sea2_ppo_v0.1.1"
type = "PPO"
save_path = "models/"

[architecture]
board_width = 6
board_length = 6

[architecture.actor]
actor_hidden_layers = [612, 256]

[architecture.critic]
critic_hidden_layers = [645, 256]

[hyperparameters]
learning_rate = 0.00002
gamma = 0.99
invalid_action_limit = 10
K_epochs = 3
clip_epsilon = 0.15

[reward_function]
no_hit = -0.05
ship_hit = 0.05
free_hit = 0.1
ship_sunk = 0.05
free_ship_sunk = 0.1
invalid_guess = -1.5
game_over = 0.1

[training]
num_episodes = 80000
```


