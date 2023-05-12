import argparse
import toml
import torch
import sys
sys.path.append('/home/nech/projects/python_projects/sea_battle_ai')  
from src.sea2_ddqn import DDQNAgent
from src.sea2_dqn import DQNAgent


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

# Run the DDQNAgent model and pass the model_save_path
agent.run()
agent.close(args.save_path)
