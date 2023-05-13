import argparse
import toml
import torch
import sys
import os

# Get the root directory of your project
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the sys.path
sys.path.append(root_dir)

from sea_ddqn import DDQNAgent
from sea_dqn import DQNAgent

# Rest of the script...



# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the TOML configuration file")
parser.add_argument('--save_path', type=str, required=True, help="Path to the directory where the trained model will be saved")
args = parser.parse_args()

# Load the TOML configuration file
config = toml.load(args.config)

# Create the DDQNAgent instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config['model']['type'] == 'ddqn':
    agent = DDQNAgent(config, device)
elif config['model']['type'] == 'dqn':
    agent = DQNAgent(config, device)
else:
    raise ValueError('Unknown model type')
# Run the DDQNAgent model and pass the model_save_path
agent.run()
agent.close(args.save_path)
