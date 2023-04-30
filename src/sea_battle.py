import random
import unittest
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque




class SeaBattleBoard:
    def __init__(self):
        """Initialize the Sea Battle board and place ships randomly."""
        self.board_size: int  = 7
        self.ships: list[dict[str, any]] = [
            {"size": 4, "location": [], "hits": 0},
            {"size": 3, "location": [], "hits": 0},
            {"size": 2, "location": [], "hits": 0},
            {"size": 1, "location": [], "hits": 0}
        ]
        self.board: list[list[str]] = [['.' for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.steps: int = 0 # Used to track the number of steps taken.
        self.place_ships_randomly()

    def place_ships_randomly(self):
        """Place ships randomly on the board."""
        for ship in self.ships:
            placed = False
            while not placed:
                row = random.randint(0, self.board_size - 1)
                col = random.randint(0, self.board_size - 1)
                orientation = random.choice(['horizontal', 'vertical'])

                if self.is_valid_location(row, col, ship["size"], orientation):
                    ship["location"] = self.place_ship(row, col, ship["size"], orientation)
                    placed = True

    def is_valid_location(self, row, col, ship_size, orientation):
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
            if col + ship_size > self.board_size:
                return False
            for i in range(col, col + ship_size):
                if not self.is_empty(row, i):
                    return False
        else:
            if row + ship_size > self.board_size:
                return False
            for i in range(row, row + ship_size):
                if not self.is_empty(i, col):
                    return False
        return True

    def is_empty(self, row, col):
        """
        Check if the given square is empty.
        
        Args:
        row (int): The row of the square.
        col (int): The column of the square.
        
        Returns:
        bool: True if the square is empty, False otherwise.
        """
        if self.board[row][col] == '.':
            return True
        return False

    def place_ship(self, row, col, ship_size, orientation):
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
        ship_location: list[tuple[int, int]] = []
        if orientation == 'horizontal':
            for i in range(col, col + ship_size):
                self.board[row][i] = 'S'
                ship_location.append((row, i))
        else:
            for i in range(row, row + ship_size):
                self.board[i][col] = 'S'
                ship_location.append((i, col))
        return ship_location

    def attack_square(self, row, col):
        """
        Attack the square at the given row and column.
        
        Args:
        row (int): The row of the square to attack.
        col (int): The column of the square to attack.
        
        Returns:
        str: 'hit', 'miss', 'sunk', or 'invalid' depending on the result of the attack.
        """
        if self.board[row][col] == 'S':
            self.board[row][col] = 'X'
            for ship in self.ships:
                if (row, col) in ship["location"]:
                    ship["hits"] += 1
                    if self.is_ship_sunk(ship):
                        self.reveal_surrounding_squares(ship)
                        return "sunk"
                    else:
                        return "hit"
        elif self.board[row][col] == '.':
            self.board[row][col] = 'M'
            return "miss"
        else:
            return "invalid"

    def reveal_surrounding_squares(self, ship: dict):
        """
        Reveal the surrounding squares of a sunken ship on the board.
        
        Args:
        ship (dict): The sunken ship, represented as a dictionary with keys 'size', 'location', and 'hits'.
        """
        for row, col in ship["location"]:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r, c = row + dr, col + dc
                    if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == '.':
                        self.board[r][c] = 'M'
    
    def is_ship_sunk(self, ship):
        """
        Check if the given ship is sunk.
        
        Args:
        ship (dict): The ship to check, represented as a dictionary with keys 'size', 'location', and 'hits'.
        
        Returns:
        bool: True if the ship is sunk, False otherwise.
        """
        return ship["hits"] == ship["size"]

    def is_game_over(self):
        """
        Check if the game is over (all ships are sunk).
        
        Returns:
        bool: True if the game is over, False otherwise.
        """
        for ship in self.ships:
            if not self.is_ship_sunk(ship):
                return False
        return True
    
    def get_step(self):
        """
        Get the step of the game.
        
        Returns:
        int: The step of the game.
        """
        return self.steps
    
    def num_unexpolored_squares(self):
        """
        Get the number of unexplored squares on the board.
        
        Returns:
        int: The number of unexplored squares on the board.
        """
        return self.board.count('.')

class SeaBattleGame:
    def __init__(self):
        """Initialize the Sea Battle game."""
        self.board = SeaBattleBoard()
        self.game_over = False

    def start_game(self, model=None, device=None):
        """Start the game and execute the main game loop."""
        print("Welcome to Sea Battle!")
        self.print_board()

        while not self.game_over:
            if model is not None and device is not None:
                state = self._get_observation().flatten()
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action_values = model(state_tensor.unsqueeze(0))
                action_values_reshaped = action_values.reshape(7, 7)
                action_values_numpy = action_values_reshaped.cpu().detach().numpy()
                x_norm = (action_values_numpy-np.min(action_values_numpy))/(np.max(action_values_numpy)-np.min(action_values_numpy))
                for row in x_norm:
                        print("\t".join("{:.2f}".format(i) for i in row))
            row, col = self.get_player_move()
            result = self.board.attack_square(row, col)
            self.print_result(result)
            self.print_board()
            self.check_game_over()

        print("Congratulations! You have sunk all the ships!")

    def print_board(self):
        """
        Print the current state of the board to the console.
        """
        print("  " + " ".join(str(i) for i in range(self.board.board_size)))
        for i, row in enumerate(self.board.board):
            print(str(i) + " " + " ".join(cell if cell != "S" else "." for cell in row))

    def _get_observation(self):
        observation = np.zeros((self.board.board_size, self.board.board_size), dtype=np.uint8)
        for i, row in enumerate(self.board.board):
            for j, cell in enumerate(row):
                if cell == '.':
                    observation[i, j] = 0
                elif cell == 'S':
                    observation[i, j] = 1
                elif cell == 'M':
                    observation[i, j] = 2
                elif cell == 'X':
                    observation[i, j] = 3
        return observation
    
    def get_player_move(self):
        """
        Get the player's move (row and column) from the console.
        
        Returns:
        tuple: A tuple (row, col) representing the player's move.
        """
        valid_move = False
        while not valid_move:
            move = input("Enter your move (row, col): ")
            row, col = self.parse_move(move)
            if row is not None and col is not None:
                valid_move = True
            else:
                print("Invalid move. Please enter a valid move.")
        return row, col

    def parse_move(self, move):
        """
        Parse the move string entered by the player.
        
        Args:
        move (str): The move string entered by the player.
        
        Returns:
        tuple: A tuple (row, col) if the move is valid, (None, None) otherwise.
        """
        try:
            row, col = [int(x) for x in move.split(',')]
            if 0 <= row < self.board.board_size and 0 <= col < self.board.board_size:
                return row, col
        except ValueError:
            pass
        return None, None

    def print_result(self, result):
        """
        Print the result of the player's move to the console.
        
        Args:
        result (str): The result of the player's move ('hit', 'miss', 'sunk', or 'invalid').
        """
        if result == "hit":
            print("Hit!")
        elif result == "miss":
            print("Miss!")
        elif result == "sunk":
            print("Sunk!")
        elif result == "invalid":
            print("Invalid move. Please try again.")

    def check_game_over(self):
        """
        Check if the game is over and set the game_over attribute accordingly.
        """
        if self.board.is_game_over():
            self.game_over = True

class UnrevealedSquaresSpace(gym.Space):
    def __init__(self, board):
        self.board = board
        super().__init__(shape=None, dtype=np.int)

    def sample(self):
        unrevealed_squares = []
        for i, row in enumerate(self.board.board):
            for j, cell in enumerate(row):
                if cell not in ['M', 'X']:
                    unrevealed_squares.append((i, j))

        if unrevealed_squares:
            selected_square = np.random.choice(len(unrevealed_squares))
            row, col = unrevealed_squares[selected_square]
            return row * self.board.board_size + col
        else:
            raise ValueError("No unrevealed squares left")

    def contains(self, x):
        return 0 <= x < self.board.board_size * self.board.board_size


class SeaBattleEnvironment(gym.Env):
    def __init__(self):
        super(SeaBattleEnvironment, self).__init__()

        self.board = SeaBattleBoard()

        # Define action and observation spaces
        self.action_space = UnrevealedSquaresSpace(self.board)
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.board.board_size, self.board.board_size), dtype=np.uint8)

    def step(self, action):
        row, col = divmod(action, self.board.board_size)
        result = self.board.attack_square(row, col)
        
        if result == 'hit':
            reward = 0
        elif result == 'sunk':
            reward = 0
        elif result == 'miss':
            # unexplored_squares = self.board.num_unexpolored_squares()
            # reward = 1.00012**(22*unexplored_squares)-1
            reward = -0.01
        else:
            reward = -1

        done = self.board.is_game_over()
        
        if done:
            reward = 1
        
        observation = self._get_observation()
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.board = SeaBattleBoard()
        return self._get_observation()

    def _get_observation(self):
        observation = np.zeros((self.board.board_size, self.board.board_size), dtype=np.uint8)
        for i, row in enumerate(self.board.board):
            for j, cell in enumerate(row):
                if cell == '.':
                    observation[i, j] = 0
                elif cell == 'S':
                    observation[i, j] = 1
                elif cell == 'M':
                    observation[i, j] = 2
                elif cell == 'X':
                    observation[i, j] = 3
        return observation

    def render(self, mode='human'):
        # You can implement a rendering method here if you want to visualize the game during training
        print("  " + " ".join(str(i) for i in range(self.board.board_size)))
        for i, row in enumerate(self.board.board):
            print(str(i) + " " + " ".join(cell if cell != "S" else "." for cell in row))
            
    def get_random_valid_action(self):
        """
            Get a random valid action (index of an unrevealed square).

            Returns:
            int: The index of an unrevealed square.
        """
        unrevealed_squares = []
        for i, row in enumerate(self.board.board):
            for j, cell in enumerate(row):
                if cell not in ['M', 'X']:
                    unrevealed_squares.append((i, j))

        if unrevealed_squares:
            selected_square = np.random.choice(len(unrevealed_squares))
            row, col = unrevealed_squares[selected_square]
            return row * self.board.board_size + col
        else:
            raise ValueError("No unrevealed squares left")
