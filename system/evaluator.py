import numpy as np
import torch
import random

from transformers import is_torch_available
from user import UserAgent
from agent import ConversationalAgent
from data import DataManager

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate_offline_trainer():
    pass

def evaluate_online_checker():
    pass