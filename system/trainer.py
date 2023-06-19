import torch

from typing import List
from torch import nn, Tensor
from torch.nn import LSTM, GRU
from base import MLP, BatchNormTrans, FM, DeepFM, PNN, ESMM, ESMM2, MMoE, DIN


class BaseTrainer(nn.Module):
    def __init__(self, feature_num: int, input_size: int):
        super.__init__()
        self.embedding = nn.Embedding(feature_num, input_size)
        self.model = None
        self.requires_mlp = True
    
    def _load_model(self):
        raise NotImplementedError

    def foward(self, x: Tensor, label_size: int):
        raise NotImplementedError


class TowerTrainer(BaseTrainer):
    def __init__(self, feature_num: int, input_size: int, hidden_sizes: List[int], 
                    dropout: float = 0.5, model_name: str = "fm"): 
        super(BaseTrainer, self).__init__(feature_num=feature_num, input_size=input_size)
        self.model_name = model_name
        self.mlp = MLP(in_channels=input_size, hidden_sizes=hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)
        self.model = self._load_model(model_name = model_name, input_size=input_size, hidden_sizes=hidden_sizes, 
                                        feature_num=feature_num, dropout=dropout)
    
    def _load_model(self, model_name, input_size, hidden_sizes, feature_num, dropout):
        if model_name in ["fm", "deepfm", "pnn"]:
            return self._load_fm_model(model_name=model_name, input_size=input_size, feature_num=feature_num)
        elif model_name in ["esmm", "esmm2", "mmoe"]:
            return self._load_expert_model(model_name=model_name, input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
        else:
            print(f"{model_name} must be in [fm, deepfm, pnn, esmm, esmm2, mmoe]")
            raise NotImplementedError

    def _load_fm_model(self, model_name, input_size, feature_num):
        w1 = nn.Embedding(feature_num, 1)
        if model_name == "fm":
            self.requires_mlp = True
            return FM(w1=w1, v=self.embedding, feature_num=feature_num, point_dot=False)
        elif model_name == "deepfm":
            self.requires_mlp = False
            fm_part = FM(w1=w1, v=self.embedding, feature_num=feature_num, point_dot=True)
            return DeepFM(fm_part=fm_part, deep_part=self.mlp)
        elif model_name == "pnn":
            self.requires_mlp = True
            fm_part = FM(w1=w1, v=self.embedding, feature_num=feature_num, point_dot=False)
            return PNN(input_size=input_size, fm_part=fm_part)
        else:
            raise NotImplementedError

    def _load_expert_model(self, model_name, input_size, hidden_sizes, dropout):
        if model_name == "esmm":
            self.requires_mlp = False
            return ESMM(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
        elif model_name == "esmm2":
            self.requires_mlp = False
            return ESMM2(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
        elif model_name == "mmoe":
            self.requires_mlp = True
            return MMoE(input_size=input_size, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x: Tensor, label_size: int):
        if self.model_name in ["esmm", "esmm2", "mmoe"]:
            x = self.embedding(x[:, -label_size:]).mean(-2)
            x = self.model(x)  # (B, L, H)/(B, L)
        else:
            x = self.model(x[:, -label_size:])  # (B, L, F)

        if self.requires_mlp:
            x = self.mlp(x)
        return x.squeeze(-1).sigmoid()


class SequenceTrainer(BaseTrainer):
    def __init__(self, feature_num: int, input_size: int, hidden_size: int, 
                    pre_hidden_sizes: List[int], dropout: float = 0.5, model_name: str = "lstm"): 
        super(BaseTrainer, self).__init__(feature_num=feature_num, input_size=input_size)
        self.mlp = MLP(input_size=hidden_size, hidden_sizes=pre_hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)
        self.model = self._load_model(model_name=model_name, input_size=input_size, dropout=dropout)
    
    def _load_model(self, model_name, input_size, hidden_size, dropout):
        if model_name == "lstm":
            return LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_name == "gru":
            return GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_name == "din":
            return DIN(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        else:
            print(f"{model_name} must be in [lstm, gru, din]")
            raise NotImplementedError

    def forward(self, x: Tensor, label_size: int):
        x = self.embedding(x).mean(-2)
        x = self.model(x)
        x = x[0] if isinstance(x, tuple) else x
        x = self.mlp(x[:, -label_size:])
        return x.squeeze(-1).sigmoid()


class GraphTrainer(BaseTrainer):
    def __init__(self, feature_num: int, input_size: int):
        super(BaseTrainer, self).__init__(feature_num=feature_num, input_size=input_size)
        raise NotImplementedError