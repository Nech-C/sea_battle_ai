import argparse
import toml
import torch
import sys
sys.path.append('/home/nech/projects/python_projects/sea_battle_ai')
import src.sea_battle_pg  # Assuming you saved the policy gradient code as sea_battle_pg.py

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the TOML configuration file")
parser.add_argument('--save_path', type=str, required=True, help="Path to the directory where the trained model will be saved")
args = parser.parse_args()

# Load the TOML configuration file
config = toml.load(args.config)

# Extract the dictionaries for architecture, hyperparameters, and training
# architecture = config["architecture"]
# hyperparameters = config["hyperparameters"]
# training = config["training"]
# model_name = config['model_name']
# reward_function = config['reward_function']

# Create the PolicyGradient instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_gradient_agent = src.sea_battle_pg.PolicyGradient(config, device)

# Run the PolicyGradient model and pass the model_save_path
policy_gradient_agent.run(args.save_path)
policy_gradient_agent.close()
