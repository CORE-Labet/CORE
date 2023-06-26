import numpy as np
import torch
import random

from user import UserAgent
from agent import ConversationalAgent
from data import DataManager
from trainer import TowerTrainer, SequenceTrainer

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_trainer(args):
    if args.trainer in ["fm", "deepfm", "pnn", "esmm", "esmm2", "mmoe"]:
        trainer = TowerTrainer(

        )
    elif args.trainer in ["lstm", "gru", "din"]:
        trainer = SequenceTrainer(
            num_feat=args.num_feat
        )


def evaluate_offline_trainer():
    pass

def evaluate_online_checker():
    pass