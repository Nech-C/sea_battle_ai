import argparse
import toml
import torch
import sys
import os
import torch.nn
import tkinter as tk

# Get the root directory of your project
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root directory to the sys.path
sys.path.append(root_dir)

from src.sea_ddqn import DDQNAgent
from src.sea_dqn import DQNAgent, DQN
from src.sea_battle import SeaBattleEnv, Observation, GuessResult, OBSERVATION_DICT


def load_model(model_path):
    model: torch.nn.Module = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


def evaluate_random_policy(env: SeaBattleEnv, num_games=1000):
    """evaluate the performance of a random policy by call the Observation.get_random_valid_action.
        the function prints the average reward,  the average effective guesses, and the average total steps

    Args:
        env (SeaBattleEnv): the SeaBattle game env                                                                        ffd
        num_games (int, optional): the number of games to be played Defaults to 100.
    """
    total_rewards = 0
    total_effective_guesses = 0
    total_steps = 0

    for _ in range(num_games):
        obs = env.reset()
        done = False

        while not done:
            action = obs.get_random_valid_action()
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs

        step_count, tot_guesses, effective_guesses, tot_reward = env.get_stats()
        total_rewards += tot_reward
        total_effective_guesses += effective_guesses
        total_steps += step_count

    avg_reward = total_rewards / num_games
    avg_effective_guesses = total_effective_guesses / num_games
    avg_steps = total_steps / num_games

    print(
        f"Random policy: \nAverage Reward: {avg_reward}\nAverage Effective Guesses: {avg_effective_guesses}\nAverage Steps: {avg_steps}")


def evaluate_agent(env: SeaBattleEnv, model: torch.nn.Module, config, num_games=1000):
    total_rewards = 0
    total_effective_guesses = 0
    total_steps = 0

    for _ in range(num_games):
        obs = env.reset()
        done = False

        while not done:
            state = obs.get_one_hot_encoding().flatten()
            state = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
            action = torch.argmax(model(state)).item()
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs

        step_count, tot_guesses, effective_guesses, tot_reward = env.get_stats()
        total_rewards += tot_reward
        total_effective_guesses += effective_guesses
        total_steps += step_count

    avg_reward = total_rewards / num_games
    avg_effective_guesses = total_effective_guesses / num_games
    avg_steps = total_steps / num_games

    print(
        f"Agent performance: \nAverage Reward: {avg_reward}\nAverage Effective Guesses: {avg_effective_guesses}\nAverage Steps: {avg_steps}")


import tkinter as tk


def create_square(canvas, x, y, size, text, fill_color):
    canvas.create_rectangle(x, y, x + size, y + size, fill=fill_color, outline='black')
    canvas.create_text(x + size // 2, y + size // 2, text=text, font=('Arial', 12))


def display_board_gui(canvas, board, q_values, width, cell_size):
    for i in range(width):
        for j in range(width):
            cell = board[i, j]
            q_value = f"{q_values[i * width + j]:.2f}"
            if cell == OBSERVATION_DICT['unknown']:
                fill_color = "white"
            elif cell == OBSERVATION_DICT['hit']:
                fill_color = "red"
            else:  # cell == Observation.MISS
                fill_color = "blue"

            x, y = j * cell_size, i * cell_size
            create_square(canvas, x, y, cell_size, q_value, fill_color)


def on_canvas_click(event, env, obs, model, config, root, canvas, width, cell_size):
    col, row = event.x // cell_size, event.y // cell_size
    action = row * width + col

    next_obs, reward, done, _ = env.step(action)
    obs = next_obs

    board = obs.get_board()
    state = obs.get_one_hot_encoding().flatten()
    state = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
    q_values = model(state).detach().cpu().numpy().flatten()

    canvas.delete('all')
    display_board_gui(canvas, board, q_values, width, cell_size)

    if done:
        print("Game over.")
        print(obs)
        root.destroy()


def user_play(env: SeaBattleEnv, model, config):
    print("Starting a new game. Type 'quit' at any time to exit.")
    obs = env.reset()
    done = False

    width = config['architecture']['board_width']
    cell_size = 50

    root = tk.Tk()
    root.title("Sea Battle")
    canvas = tk.Canvas(root, width=width * cell_size, height=width * cell_size, bg='white')
    canvas.pack()

    board = obs.get_board()
    state = obs.get_one_hot_encoding().flatten()
    state = torch.from_numpy(state).float().unsqueeze(0).to("cuda")
    q_values = model(state).detach().cpu().numpy().flatten()

    display_board_gui(canvas, board, q_values, width, cell_size)

    canvas.bind("<Button-1>",
                lambda event: on_canvas_click(event, env, obs, model, config, root, canvas, width, cell_size))
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument('--config', type=str, required=True, help='Path to game configuration.')
    args = parser.parse_args()

    # Load model and configuration
    config = toml.load(args.config)
    hyperparameters = config['hyperparameters']
    architecture = config['architecture']
    model = config['model']
    model = load_model(args.model_path, model['type'])

    shape = (config['architecture']['board_length'], config['architecture']['board_width'])
    env = SeaBattleEnv(board_shape=shape)
    while True:
        print("1. evaluate agent\n2. evaluate random policy\n3. play Sea Battle with agent assistance")
        choice = int(input("Choose an option (1, 2, or 3): "))
        if choice == 1:
            evaluate_agent(env, model, config)
        elif choice == 2:
            evaluate_random_policy(env)
        elif choice == 3:
            user_play(env, model, config)
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")


if __name__ == '__main__':
    main()
