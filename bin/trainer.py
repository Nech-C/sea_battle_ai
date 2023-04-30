import argparse
import toml
import torch
import sys
sys.path.append('/home/nech/projects/python_projects/sea_battle_ai')
import src.sea_battle_ddqn

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the TOML configuration file")
parser.add_argument('--save_path', type=str, required=True, help="Path to the directory where the trained model will be saved")
args = parser.parse_args()

# Load the TOML configuration file
config = toml.load(args.config)

# Extract the dictionaries for architecture, hyperparameters, and training
architecture = config["architecture"]
hyperparameters = config["hyperparameters"]
training = config["training"]
model_name = config['model_name']
# Create the DoubleDQN instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
double_dqn = src.sea_battle_ddqn.DoubleDQN(architecture, hyperparameters, training, device,model_name)

# Run the DoubleDQN model and pass the model_save_path
double_dqn.run(args.save_path)
