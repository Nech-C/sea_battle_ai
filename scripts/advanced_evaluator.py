import argparse

import numpy as np
import toml
import torch
import sys
import os
import torch.nn

# Get the root directory of your project
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the root directory to the sys.path
sys.path.append(root_dir)

from src.sea_battle import SeaBattleEnv, Observation

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_models(model_paths: list):
    models = []
    for model_path in model_paths:
        models.append(torch.load(model_path).eval())
    return models


def eval_model(env: SeaBattleEnv, model: torch.nn.Module, num_games: int):
    '''

    Args:
        model: the model to be tested
        num: the number of games to test

    Returns: the results of the evaluation

    '''
    total_step = 0
    total_guesses = 0
    total_effective_guesses = 0
    total_reward = 0
    total_stuck = 0

    for i in range(num_games):
        obs: Observation = env.reset()
        done = False
        # print("run once")
        while not done:
            state: np.ndarray = obs.get_one_hot_encoding().flatten()
            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            action = torch.argmax(model(state)).item()
            next_obs, _, done, _ = env.step(action)
            # print(f"i: {i}, action: {action}")
            # print(env.get_stats())
            obs = next_obs
            step_cnt, _, _, _ = env.get_stats()
            if step_cnt > len(obs.get_one_hot_encoding().flatten()):
                total_stuck += 1
                break

        step_count, guess_count, effective_guess_count, reward = env.get_stats()
        total_step += step_count
        total_guesses += guess_count
        total_effective_guesses += effective_guess_count
        total_reward += reward

    average_step = total_step / num_games
    average_guesses = total_guesses / num_games
    average_effective_guesses = total_effective_guesses / num_games
    average_reward = total_reward / num_games
    average_stuck = total_stuck / num_games

    return average_step, average_guesses, average_effective_guesses, average_reward, average_stuck


def main():
    # read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_config', type=str, required=True, help='Path to test config')
    args = parser.parse_args()

    # load config and models
    config = toml.load(args.test_config)
    models = load_models(config['models'])
    print(models)
    env = SeaBattleEnv(board_shape=(config["board_length"], config["board_width"]))
    for model in models:
        avg_step, avg_guesses, avg_eff_guesses, avg_reward, avg_stuck = eval_model(env, model, config["num_games"])
        print(model)
        print(f"average step: {avg_step: .4f} average number of guesses: {avg_guesses: .4f} average number of effective"
              f": {avg_eff_guesses: .4f} average reward: {avg_reward: .2f} "
              f"average number of being stuck: {avg_stuck: .2f}")


if __name__ == "__main__":
    main()
