# sea_battle.py:
import numpy as np
import gym
import numpy.random as npr
from enum import Enum
from typing import (Any, Dict, List, Tuple)

CELL_MAPPING={'empty': 0, 'ship': 1, 'miss': 2, 'hit': 3}
RENDER_MAPPING={0: '.', 1: '.', 2: 'M', 3:'X'}
OBSERVATION_MAPPING={0: 0, 1: 0, 2: 2, 3:3}
OBSERVATION_DICT={"unknown": 0, 'miss': 2, 'hit': 3}

class GuessResult(Enum):
    NO_HIT = 1
    SHIP_HIT = 2
    FREE_HIT = 3
    SHIP_SUNK = 4
    FREE_SHIP_SUNK = 5
    INVALID_GUESS = 6
    GAME_OVER = 7

REWARD_MAPPING = {
    GuessResult.NO_HIT: 0,
    GuessResult.SHIP_HIT: 0,
    GuessResult.FREE_HIT: 0.5,
    GuessResult.SHIP_SUNK: 0,
    GuessResult.FREE_SHIP_SUNK: 0.5,
    GuessResult.INVALID_GUESS: -1.0,
    GuessResult.GAME_OVER: 0.5
}


class Observation():
        def __init__(self, board: np.ndarray):
            self.board = board
            self.board[self.board == CELL_MAPPING['empty']] = OBSERVATION_MAPPING[CELL_MAPPING['empty']]
            self.board[self.board == CELL_MAPPING['ship']] = OBSERVATION_MAPPING[CELL_MAPPING['ship']]
            self.board[self.board == CELL_MAPPING['miss']] = OBSERVATION_MAPPING[CELL_MAPPING['miss']]
            self.board[self.board == CELL_MAPPING['hit']] = OBSERVATION_MAPPING[CELL_MAPPING['hit']]
            
        def get_one_hot_encoding(self):
            one_hot_board = np.zeros((*self.board.shape, 3))
            
            for i in range(one_hot_board.shape[0]):
                for j in range(one_hot_board.shape[1]):
                    square = self.board[i, j]
                    if square == OBSERVATION_DICT['unknown']:
                        one_hot_board[i, j, 0] = 1
                    elif square == OBSERVATION_DICT['miss']:
                        one_hot_board[i, j, 1] = 1
                    elif square == OBSERVATION_DICT['hit']:
                        one_hot_board[i, j, 2] = 1
            
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
    def __init__(self, shape=(7,7), reward_func=REWARD_MAPPING):
        super(SeaBattleEnv, self).__init__()
        self.shape = shape
        self.reset()
        self.reward_function = reward_func
    
    def observation(self):
        return Observation(self.board.copy())
        
    def step(self, action: int):
        """advance the game

        Args:
            action (int): the number of square to guess. 
            0   1   2   3   4   5   6   7
            8   9   10  11  12  13  14  15
            .
            .
            .
        """
        row, col = divmod(action, self.board.shape[1])
        result = self.attack_square(row, col)
        
        self.step_count += 1
        if result != GuessResult.INVALID_GUESS:
            self.tot_guesses += 1
            if result != GuessResult.FREE_SHIP_SUNK and result != GuessResult.FREE_HIT and result != GuessResult.GAME_OVER:
                self.effective_guesses += 1
        
        info = {
            'ships': np.count_nonzero(self.board == CELL_MAPPING['ship'])
        }
        self.tot_reward += self.reward_function[result]
        return (self.observation(), self.reward_function[result], result == GuessResult.GAME_OVER, info)
        
    def reset(self):
        self.step_count = 0
        self.tot_guesses = 0
        self.effective_guesses = 0
        self.tot_reward = 0
        self.board: np.ndarray = np.full(self.shape, CELL_MAPPING['empty'])
        self.ships: List[Dict[str, Any]] = [
            {"size": 4, "location": [], "hits": 0},
            {"size": 3, "location": [], "hits": 0},
            {"size": 2, "location": [], "hits": 0},
            {"size": 1, "location": [], "hits": 0}
        ]
        self.place_ships_randomly()
        self.last_guess_correct = False
    
        return self.observation()

    def render(self):
        print("  " + " ".join(str(i) for i in range(self.board.shape[1])))
        for i, row in enumerate(self.observation().get_board()):
            print(str(i) + " " + " ".join(RENDER_MAPPING[cell] for cell in row))
    
    def close(self):
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
            else:
                if row + ship_size > self.board.shape[0]:
                    return False
                for i in range(row, row + ship_size):
                    if not self.board[i, col] == CELL_MAPPING['empty']:
                        return False
            return True
    
    def place_ship(self, row: int, col: int, ship_size: int, orientation: str):
        """
        Place the ship on the board with the given row, col, ship_size, and orientation.
        
        Args:
        row (int): The starting row of the ship.
        col (int): The starting column of the ship.
        ship_size (int): The size of the ship.
        orientation (str): The orientation of the ship ('horizontal' or 'vertical').
        
        Returns:
        list: A list of tuples representing the ship's location.
        """
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
                row = npr.randint(0, self.board.shape[0])
                col = npr.randint(0, self.board.shape[1])
                orientation = npr.choice(['horizontal', 'vertical'])

                if self.is_valid_location(row, col, ship["size"], orientation):
                    ship["location"] = self.place_ship(row, col, ship["size"], orientation)
                    placed = True
    
    def is_game_over(self) -> bool:
        return all(ship["hits"] == len(ship["location"]) for ship in self.ships)
    
    def is_ship_sunk(self, ship):
        """
        Check if the given ship is sunk.
        
        Args:
        ship (dict): The ship to check, represented as a dictionary with keys 'size', 'location', and 'hits'.
        
        Returns:
        bool: True if the ship is sunk, False otherwise.
        """
        return ship["hits"] == ship["size"]
    
    def reveal_surrounding_squares(self, ship: dict):
        """
    Reveal the surrounding squares of a sunken ship on the board.
    b
    Args:
    ship (dict): The sunken ship, represented as a dictionary with keys 'size', 'location', and 'hits'.
    """
        for row, col in ship["location"]:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == CELL_MAPPING['empty']:
                        self.board[r, c] = CELL_MAPPING['miss']
    
    def attack_square(self, row: int, col: int):
        """
            Attack the square at the given row and column.

            Args:
            row (int): The row of the square to attack.
            col (int): The column of the square to attack.

            Returns:
            GuessResult: An Enum member representing the result of the attack: NO_HIT, SHIP_HIT, FREE_HIT, SHIP_SUNK, FREE_SHIP_SUNK, INVALID_GUESS,  GAME_OVER.
        """
        if row < 0 or row >= self.board.shape[0] or col < 0 or col >= self.board.shape[1]:
            return GuessResult.INVALID_GUESS
        
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
            return GuessResult.INVALID_GUESS
        
    def __str__(self):
        board_str = "  " + " ".join(str(i) for i in range(self.board.shape[1])) + "\n"
        for i, row in enumerate(self.board):
            board_str += str(i) + " " + " ".join(RENDER_MAPPING[cell] for cell in row) + "\n"
        return board_str

    def get_stats(self)->tuple: 
        """return the current stats of the env

        Returns:
            tuple: (step_count, total num of guesses, num of effective guesses, total reward)
        """
        return (self.step_count, self.tot_guesses, self.effective_guesses, self.tot_reward)

if __name__ == "__main__":
    shape = (4,4)
    env = SeaBattleEnv(shape)
    for step in range(200):
        env.render()
        action = int(input("Enter action:"))
        action = env.observation().get_random_valid_action()
        obs, reward, done, info = env.step(action)

        print("----------result:------------")
        print("obs:")
        print(obs.get_board())
        print("obs_one_hot:")
        print(obs.get_one_hot_encoding().flatten())
        step, tot_guesses, effective_guesses = env.get_stats()
        print("stats:")
        print(f"reward: {reward}")
        print(f"step: {step}")
        print(f"tot_guesses: {tot_guesses}")
        print(f"effective_guesses: {effective_guesses}")
        print("---------------------------------")

        if done:
            print("Game over!")
            env.reset()
    env.close()
        