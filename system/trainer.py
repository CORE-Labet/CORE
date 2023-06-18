import torch

from typing import List
from torch import nn
from base import MLP, BatchNormTrans


class BaseTrainer(nn.Module):
    def __init__(self, feature_num: int, input_size: int, hidden_sizes: List[int], dropout: float = 0.5):
        super.__init__()
        self.embedding = nn.Embedding(feature_num, input_size)
        self.mlp = MLP(in_channels=input_size, hidden_sizes=hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)
        self.model = None
        self.requires_mlp = True
    
    def load(self):
        raise NotImplementedError

    def foward(self, label_size: int):
        raise NotImplementedError


class TowerTrainer(BaseTrainer):
    pass


class SequenceTrainer(BaseTrainer):
    pass


class GraphTrainer(BaseTrainer):
    pass