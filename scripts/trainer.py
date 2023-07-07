import argparse
import toml
import torch
import sys
import os

# Get the root directory of your project
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the sys.path
sys.path.append(root_dir)

from src.sea_ddqn import DDQNAgent
from src.sea_dqn import DQNAgent
from src.sea_a2c import A2C
from src.sea_ppo import PPO
# Rest of the script...



# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the TOML configuration file")
args = parser.parse_args()

# Load the TOML configuration file
config = toml.load(args.config)

# Create the rl training instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_name = config['model']['type']
class_obj = globals()[class_name]
training = class_obj(config, device)

# Run the rl model and pass the model_save_path
training.run()
training.close(config['model']['save_path'])
