"""Definition of training constants, and other things which do not need configuration."""
import torch
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