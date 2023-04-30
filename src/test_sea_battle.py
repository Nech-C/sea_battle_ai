import unittest
from sea_battle import SeaBattleBoard, SeaBattleEnvironment, SeaBattleGame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
from sea_battle_dqn import DeepQNetwork

def has_right_ship_count(board: SeaBattleBoard) -> bool:
    squares_occupied = set()
    ships = [1, 2, 3, 4]
    for ship in board.ships:
        size = 0
        for square in ship["location"]:
            size += 1
            if square in squares_occupied:
                return False
            else:
                squares_occupied.add(square)
        try:
            ships.remove(size)
        except ValueError:
            return False
    if len(ships) != 0:
        return False
    return True

def get_adjacent_squares(square):
    row, col = square
    adjacent_squares = []

    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 10:
                adjacent_squares.append((new_row, new_col))

    return adjacent_squares


def no_adjacent_ships(board: SeaBattleBoard) -> bool:
    for ship in board.ships:
        for square in ship["location"]:
            for adjacent_square in get_adjacent_squares(square):
                # Check if the adjacent square belongs to another ship
                for other_ship in board.ships:
                    if other_ship != ship and adjacent_square in other_ship["location"]:
                        return False
    return True
   
def is_valid_board(board):
    return has_right_ship_count(board) and no_adjacent_ships(board)

def test_env():
    env = SeaBattleEnvironment()
    done = False
    tot_reward = 0
    action = 0
    while not done:
        _, reward, done, _ = env.step(action)
        env.render()
        action += 1
        tot_reward += reward
        print(f"total reward: {tot_reward}")

def test_model(hidden_layer_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork(49, device, hidden_layer_size)
    model.load_state_dict(torch.load("trained_model.pth"))

    for _ in range(3):
        game:  SeaBattleGame  = SeaBattleGame()
        game.start_game(model, device)

class TestSeaBattleBoard(unittest.TestCase):
    def test_place_ships_randomly(self):
        for _ in range(100):  # Run the test for 100 iterations
            board = SeaBattleBoard()
            board.place_ships_randomly()
            if not is_valid_board(board):
                print("Invalid board:")
                print(board.__str__)
                self.assertTrue(is_valid_board(board))
            else:
                self.assertTrue(is_valid_board(board))




if __name__ == '__main__':
    # unittest.main()
    
    test_model(hidden_layer_size=[128, 128])