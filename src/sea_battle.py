import numpy as np
import gym
import random as npr
from enum import Enum
from typing import (Any, Dict, List, Tuple)


class GuessResult(Enum):
    NO_HIT = 1
    SHIP_HIT = 2
    FREE_HIT = 3
    SHIP_SUNK = 4
    FREE_SHIP_SUNK = 5
    INVALID_GUESS = 6
    GAME_OVER = 7
    GAME_LOST = 8
    GAME_WON = 9


CELL_MAPPING = {'empty': 0, 'ship': 1, 'miss': 2, 'hit': 3}
RENDER_MAPPING = {0: '.', 1: '.', 2: 'M', 3: 'X'}
OBSERVATION_MAPPING = {0: 0, 1: 0, 2: 2, 3: 3}
OBSERVATION_MAPPING_LIST = [0, 0, 2, 3]
OBSERVATION_DICT = {"unknown": 0, 'miss': 2, 'hit': 3}
DEFAULT_CONFIG = {
    "INVALID_ACTION_RESULT": GuessResult.INVALID_GUESS,
    "seed": None,
    "ALLOW_ADJACENT_PLACEMENT": False,
    "SIZE1_NUM": 4,
    "SIZE2_NUM": 3,
    "SIZE3_NUM": 2,
    "SIZE4_NUM": 1
}
REWARD_MAPPING = {
    GuessResult.NO_HIT: 0,
    GuessResult.SHIP_HIT: 0,
    GuessResult.FREE_HIT: 0.5,
    GuessResult.SHIP_SUNK: 0,
    GuessResult.FREE_SHIP_SUNK: 0.5,
    GuessResult.INVALID_GUESS: -1.0,
    GuessResult.GAME_OVER: 0.5,
    GuessResult.GAME_LOST: -1,
    GuessResult.GAME_WON: 0.5,
}


class Observation:

    def __init__(self, board: np.ndarray):
        self.board = np.array(OBSERVATION_MAPPING_LIST)[board]

    def get_one_hot_encoding(self):
        num_classes = 3  # Number of different cell values
        one_hot_board = np.eye(num_classes)[self.board]
        return one_hot_board

    def get_board(self):
        return self.board

    def get_random_valid_action(self):
        """
            Get a random valid action.

            Returns:
            int: A random valid action."""
        valid_actions = np.where(self.board.flatten() == OBSERVATION_DICT['unknown'])
        return npr.choice(valid_actions[0])


class SeaBattleEnv(gym.Env):
    def __init__(self, board_shape=(6, 6), reward_func=REWARD_MAPPING, placement: List[Dict[str, Any]] = None,
                 config: Dict = DEFAULT_CONFIG):
        super(SeaBattleEnv, self).__init__()
        self.last_guess_correct = None
        self.ships = None
        self.board = None
        self.tot_reward = None
        self.effective_guesses = None
        self.tot_guesses = None
        self.step_count = None
        self.shape = board_shape
        self.config = config
        self.reward_function = reward_func

        # Set the seed for the environment
        if self.config['seed'] is not None:
            npr.seed(self.config['seed'])

        self.reset(placement)

    def observation(self):
        return Observation(self.board.copy())

    def step(self, action: int):
        """Advance the game by performing the given action."""

        row, col = divmod(action, self.board.shape[1])
        result = self.attack_square(row, col)

        self.step_count += 1
        if result != GuessResult.INVALID_GUESS:
            self.tot_guesses += 1
            if result != GuessResult.FREE_SHIP_SUNK and result != GuessResult.FREE_HIT and result != GuessResult.GAME_OVER:
                self.effective_guesses += 1

        extra_info = {
            'ships': np.count_nonzero(self.board == CELL_MAPPING['ship'])
        }
        self.tot_reward += self.reward_function[result]
        is_done = result in [GuessResult.GAME_OVER, GuessResult.GAME_LOST, GuessResult.GAME_WON]
        return self.observation(), self.reward_function[result], is_done, extra_info

    def reset(self, placement: List[Dict[str, Any]]=None):
        """Reset the environment to its initial state."""

        self.step_count = 0
        self.tot_guesses = 0
        self.effective_guesses = 0
        self.tot_reward = 0
        self.board: np.ndarray = np.full(self.shape, CELL_MAPPING['empty'])
        self.ships: List[Dict[str, Any]] = [
            {"size": size, "location": [], "hits": 0}
            for size in range(4, 0, -1)
            for _ in range(self.config[f"SIZE{size}_NUM"])
        ]
        self.last_guess_correct = False
        if placement is None:
            self.place_ships_randomly()
        else:
            self.load_ship_placement(placement)
        return self.observation()

    def render(self):
        """Render the current state of the game."""

        print("  " + " ".join(str(i) for i in range(self.board.shape[1])))
        for i, row in enumerate(self.observation().get_board()):
            print(str(i) + " " + " ".join(RENDER_MAPPING[cell] for cell in row))

    def close(self):
        """Close the environment."""

        pass

    def is_valid_location(self, row: int, col: int, ship_size: int, orientation: str):
        """
        Check if the given row, col, and orientation is a valid location for the ship.

        Args:
        row (int): The starting row of the ship.
        col (int): The starting column of the ship.
        ship_size (int): The size of the ship.
        orientation (str): The orientation of the ship ('horizontal' or 'vertical').

        Returns:
        bool: True if the location is valid, False otherwise.
        """
        if orientation == 'horizontal':
            if col + ship_size > self.board.shape[1]:
                return False
            for i in range(col, col + ship_size):
                if not self.board[row, i] == CELL_MAPPING['empty']:
                    return False
                if not self.config['ALLOW_ADJACENT_PLACEMENT']:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 1]:  # Only check left and right squares
                            r, c = row + dr, i + dc
                            if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == \
                                    CELL_MAPPING['ship']:
                                return False
        else:  # 'vertical'
            if row + ship_size > self.board.shape[0]:
                return False
            for i in range(row, row + ship_size):
                if not self.board[i, col] == CELL_MAPPING['empty']:
                    return False
                if not self.config['ALLOW_ADJACENT_PLACEMENT']:
                    for dr in [-1, 1]:  # Only check top and bottom squares
                        for dc in [-1, 0, 1]:
                            r, c = i + dr, col + dc
                            if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == \
                                    CELL_MAPPING['ship']:
                                return False
        return True

    def place_ship(self, row: int, col: int, ship_size: int, orientation: str):
        """Place the ship on the board at the given location and orientation."""

        ship_location: List[Tuple[int, int]] = []
        if orientation == 'horizontal':
            for i in range(col, col + ship_size):
                self.board[row, i] = CELL_MAPPING['ship']
                ship_location.append((row, i))
        else:
            for i in range(row, row + ship_size):
                self.board[i, col] = CELL_MAPPING['ship']
                ship_location.append((i, col))
        return ship_location

    def place_ships_randomly(self):
        """Place ships randomly on the board."""

        for ship in self.ships:
            placed = False
            while not placed:
                row = npr.randint(0, self.board.shape[0] - 1)
                col = npr.randint(0, self.board.shape[1] - 1)

                orientation = npr.choice(['horizontal', 'vertical'])

                if self.is_valid_location(row, col, ship["size"], orientation):
                    ship["location"] = self.place_ship(row, col, ship["size"], orientation)
                    placed = True

    def load_ship_placement(self, placement: List[Dict[str, Any]]):
        """Load the ship placement onto the board."""

        self.ships = placement
        self.board = np.full(self.shape, CELL_MAPPING['empty'])
        for ship in self.ships:
            for location in ship['location']:
                row, col = location
                self.board[row, col] = CELL_MAPPING['ship']

    def is_game_over(self) -> bool:
        """Check if the game is over."""

        return all(ship["hits"] == len(ship["location"]) for ship in self.ships)

    def is_ship_sunk(self, ship):
        """Check if the given ship is sunk."""

        return ship["hits"] == ship["size"]

    def reveal_surrounding_squares(self, ship: dict):
        """Reveal the surrounding squares of a sunken ship on the board."""

        for row, col in ship["location"]:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == \
                            CELL_MAPPING['empty']:
                        self.board[r, c] = CELL_MAPPING['miss']

    def attack_square(self, row: int, col: int):
        """Attack the square at the given row and column."""

        if row < 0 or row >= self.board.shape[0] or col < 0 or col >= self.board.shape[1]:
            return self.config["INVALID_ACTION_RESULT"]

        if self.board[row, col] == CELL_MAPPING['ship']:
            self.board[row, col] = CELL_MAPPING['hit']
            for ship in self.ships:
                if (row, col) in ship["location"]:
                    ship["hits"] += 1
                    if self.last_guess_correct:
                        guess_result = GuessResult.FREE_HIT
                    else:
                        guess_result = GuessResult.SHIP_HIT
                        self.last_guess_correct = True

                    if self.is_ship_sunk(ship):
                        self.reveal_surrounding_squares(ship)
                        if self.is_game_over():
                            return GuessResult.GAME_OVER
                        else:
                            return GuessResult.FREE_SHIP_SUNK if guess_result == GuessResult.FREE_HIT else GuessResult.SHIP_SUNK
                    else:
                        return guess_result
        elif self.board[row, col] == CELL_MAPPING['empty']:
            self.board[row, col] = CELL_MAPPING['miss']
            self.last_guess_correct = False
            return GuessResult.NO_HIT
        else:
            return self.config["INVALID_ACTION_RESULT"]

    def __str__(self):
        """Convert the current state of the game to a string."""

        board_str = "  " + " ".join(str(i) for i in range(self.board.shape[1])) + "\n"
        for i, row in enumerate(self.board):
            board_str += str(i) + " " + " ".join(RENDER_MAPPING[cell] for cell in row) + "\n"
        return board_str

    def get_stats(self):
        """Get the current stats of the game."""

        return self.step_count, self.tot_guesses, self.effective_guesses, self.tot_reward


if __name__ == "__main__":
    shape = (10, 10)
    env = SeaBattleEnv(shape)
    for step in range(100000):
        # env.render()
        action = env.observation().get_random_valid_action()
        # aaction = int(input("enter action"))
        obs, reward, done, info = env.step(action)

        # print("----------result:------------")
        # print("obs:")
        # print(obs.get_board())
        # print("obs_one_hot:")
        # print(obs.get_one_hot_encoding().flatten())
        # step, tot_guesses, effective_guesses, tot_reward = env.get_stats()
        # print("stats:")
        # print(f"reward: {reward}")
        # print(f"step: {step}")
        # print(f"tot_guesses: {tot_guesses}")
        # print(f"effective_guesses: {effective_guesses}")
        # print(f"tot_reward: {tot_reward}")
        # print("---------------------------------")

        if done:
            print("Game over!")
            env.reset()
    env.close()
