import random
import numpy as np
import toml
from collections import deque
from abc import ABC, abstractmethod
from lib.constants import CELL_MAPPING
from src.sea_battle2 import GuessResult
from typing import Dict


class ReplayBuffer():
    """A simple replay buffer implementation to store and sample experiences."""
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
    def push(self, experience: tuple):
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class Training_instance(ABC):
    def __init__(self, config, device):
        self.model_name = config["model"]["model_name"]
        self.architecture = config["architecture"]
        self.hyperparameters = config["hyperparameters"]
        self.reward_function = config["reward_function"]
        self.training = config["training"]
        self.reward_function = load_reward_function(config["reward_function"])
        self.device = device

    @abstractmethod
    def run(self):
        pass

def load_reward_function(reward_function: Dict) -> Dict[GuessResult, float]:
    # Map the keys back to the Enum members
    reward_mapping = {
        GuessResult.NO_HIT: reward_function["no_hit"],
        GuessResult.SHIP_HIT: reward_function["ship_hit"],
        GuessResult.FREE_HIT: reward_function["free_hit"],
        GuessResult.SHIP_SUNK: reward_function["ship_sunk"],
        GuessResult.FREE_SHIP_SUNK: reward_function["free_ship_sunk"],
        GuessResult.INVALID_GUESS: reward_function["invalid_guess"],
        GuessResult.GAME_OVER: reward_function["game_over"]
    }
    return reward_mapping
