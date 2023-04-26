import torch
import random
from enum import Enum

# Make cpu as torch.
DEVICE = "cuda"

class Task(Enum):
    # Worker performs training (returns: parameters)
    TRAIN = "train"
    # Worker performs evaluation (returns: reward)
    EVAL = "eval"
    # Worker performs data collection (returns: replay buffer)
    COLLECT = "collect"

    @classmethod
    def selection(cls):
        weights = [0.2, 0.1, 0.7]
        return random.choices(list(cls), weights=weights)[0]
    