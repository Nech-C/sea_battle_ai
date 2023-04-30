import random
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from lib.constants import CELL_MAPPING

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
    def __init__(self):
        pass
    @abstractmethod
    def run(self):
        pass




def state_to_one_hot_3(state: np.ndarray):
    """convert a state(2d np array) to the corresponding one-hot encoding. This function should be used with env.step()

    Args:
        state (ndarray): array with integers 0, 2, 3. 0 for empty or ship, 2 for miss, 3 for hit.

    Returns:
        ndarray: the corresponding one-hot encoding
    """
    one_hot_state = np.zeros((*state.shape, 3), dtype=np.float32)

    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            square = state[i, j]
            if square == 0:  # empty and ship
                one_hot_state[i, j, 0] = 1
            elif square == 2:  # miss
                one_hot_state[i, j, 1] = 1
            elif square == 3:  # hit
                one_hot_state[i, j, 2] = 1

    return one_hot_state.flatten()
