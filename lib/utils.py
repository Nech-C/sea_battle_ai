# utils.py:
import random
import numpy as np
import toml
import torch
import os
from collections import deque
from abc import ABC, abstractmethod
from src.sea_battle import GuessResult, SeaBattleEnv
from typing import Dict
from torch.utils.tensorboard import SummaryWriter



class ReplayBuffer():
    """A simple replay buffer implementation to store and sample experiences."""
    def __init__(self, max_size: int):
        self.buffer = deque([], maxlen=max_size)
    def push(self, experience: tuple):
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class Training_instance(ABC):
    def __init__(self, config: dict, device):
        self.model_name = config["model"]["model_name"]
        self.architecture = config["architecture"]
        self.hyperparameters = config["hyperparameters"]
        self.reward_function = config["reward_function"]
        self.training = config["training"]
        self.reward_function = load_reward_function(config["reward_function"])
        self.device = device
        self.writer = SummaryWriter(log_dir=f'./runs/{self.model_name}', flush_secs=1)

    @abstractmethod
    def run(self):
        pass
    
    def close(self, model_save_path):
        """the method saves the model to the specified path and closes the tensorboard writer

        Args:
            model_save_path (str): the string of the save path
        """
        torch.save(self.network, os.path.join(model_save_path, self.model_name + ".pth"))
        self.writer.close()
    
    def update_tensorboard(self, episode: int, env: SeaBattleEnv, loss: float, eps: float):
        """ the method updates the tensorboard

        Args:
            episode (int): the eposode number
            env (SeaBattleEnv): the SeaBattleEnv object
            loss (float): policy update loss 
            eps (float): epsilon
        """
        step_count, tot_guesses, effective_guesses, tot_reward = env.get_stats()
        if loss is not None:
            self.writer.add_scalar('Loss/train', loss, episode)
        self.writer.add_scalar('Steps/train', step_count, episode)
        self.writer.add_scalar('Guesses/train', tot_guesses, episode) 
        self.writer.add_scalar('Effective guesses/train', effective_guesses, episode)
        if eps is not None:
            self.writer.add_scalar('Epsilon/train', eps, episode)
        self.writer.add_scalar('reward/train', tot_reward, episode)
        print(f"episode: {episode}\t reward:{tot_reward}")
    

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
