import torch

from torch import nn, Tensor
from typing import List, Optional, Callable


class MLP(nn.Sequential):
    def __init__(self, input_size: int, hidden_sizes: List[int], norm_layer: Optional[Callable[..., nn.Module]] = None,
                    activation_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU, bias: bool = True, 
                    dropout: float = 0.0, for_pre=True):
        layers = []   
        
        in_dim = input_size
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(in_dim, hidden_size, bias=bias))
            if norm_layer:
                layers.append(norm_layer(hidden_size))
            layers.append(activation_layer())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, hidden_sizes[-1], bias=bias))

        if not for_pre:
            if norm_layer:
                layers.append(norm_layer(hidden_sizes[-1]))
            layers.append(activation_layer())
            layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class BatchNormTrans(nn.BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C), C is the number of features we want to norm
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ESMM(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout: float):
        super(ESMM, self).__init__()
        self.mlp_ctr = MLP(input_size=input_size, hidden_sizes=hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)
        self.mlp_cvr = MLP(input_size=input_size, hidden_sizes=hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:  # (B, L, I)
        h_ctr = torch.sigmoid(self.mlp_ctr(x))
        h_cvr = torch.sigmoid(self.mlp_cvr(x))
        return h_ctr * h_cvr


class ESMM2(ESMM):
    def __init__(self, input_size: int, hidden_sizes: int, dropout: float):
        super().__init__(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)
        self.mlp_car = MLP(input_size=input_size, hidden_sizes=hidden_sizes, norm_layer=BatchNormTrans, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        h_ctr = torch.sigmoid(self.mlp_ctr(x))
        h_car = torch.sigmoid(self.mlp_car(x))
        h_cvr = torch.sigmoid(self.mlp_cvr(x))
        return h_ctr * h_car * h_cvr


class MMoE(nn.Module):
    def __init__(self, input_size: int, dropout: float, expert_num: int = 3):
        super(MMoE, self).__init__()
        self.gate = nn.Sequential(
                        nn.Linear(in_features=input_size, out_features=expert_num),
                        nn.Softmax(dim=-1)
                    )
        self.experts = nn.ModuleList()
        for _ in range(expert_num):
            self.experts.append(MLP(input_size=input_size, hidden_sizes=[input_size, input_size], dropout=dropout, for_pre=False))

    def forward(self, x: Tensor) -> Tensor:
        h_expert = torch.stack([expert(x) for expert in self.experts], -1)  # (B, L, I, 3)
        gate_weight = self.gate(x).unsqueeze(-2)  # (B, L, 1, 3)
        return torch.sum(h_expert * gate_weight, dim=-1)  # (B, L, I)


class FM(nn.Module):
    def __init__(self, w_dim: int = 0, v_dim: int = 0, w1: nn.Embedding = None, v: nn.Embedding =None, 
                    point_dot: bool = False, num_field: int = 0, weight: float = 0.5):
        super(FM, self).__init__()
        self.W0 = nn.Parameter(torch.randn(1))
        self.W1 = w1 if w1 else nn.Embedding(w_dim, 1)  # for linear part
        self.V = v if v else nn.Embedding(w_dim, v_dim) # for product part
        self.F = nn.Parameter(torch.randn(num_field, self.V.weight.shape[1])) if num_field > 0 else None

        self.point_dot = point_dot
        self.weight = weight
    
    def forward(self, x: Tensor) -> Tensor:
        # inputs: (B, ..., F), F is the num of fields
        linear_part = torch.sum(self.W1(x), dim=-2) + self.W0  # (B, ..., 1)
        x = self.V(x)  # (B, ..., F, K)
        if self.F:
            x += self.F
        product_part = torch.pow(torch.sum(x, -2), 2)  # (B, ..., K)
        product_part += torch.sum(torch.pow(x, 2), -2)
        if self.point_dot:
            product_part = torch.mean(product_part, -1, keepdim=True)  # (B, ..., 1)
        return linear_part + self.weight * product_part


class DeepFM(nn.Module):
    def __init__(self, fm_part: FM, deep_part: MLP):
        super(DeepFM, self).__init__()
        self.fm_part = fm_part
        self.deep_part = deep_part
        self.V = fm_part.V

    def forward(self, x: Tensor) -> Tensor:
        embed_inputs = torch.mean(self.V(x), -2)
        return self.fm_part(x) + self.deep_part(embed_inputs)


class PNN(nn.Module):
    def __init__(self, input_size: int, fm_part: FM):
        super(PNN, self).__init__()
        self.fm_part = fm_part
        self.ln = nn.Linear(input_size * 2, input_size)
        self.V = fm_part.V

    def forward(self, x: Tensor) -> Tensor:
        embed_inputs = self.V(x).mean(dim=-2)
        fm_inputs = self.fm_part(x)
        embed_norm = embed_inputs.norm(dim=-1, keepdim=True, p=2)
        fm_norm = fm_inputs.norm(dim=-1, keepdim=True, p=2)
        fm_inputs = fm_inputs / fm_norm * embed_norm
        outputs = torch.cat((fm_inputs, embed_inputs), dim=-1)
        return self.ln(outputs)


class DIN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, attention_size: int = 36):
        super(DIN, self).__init__()
        self.ln = nn.Linear(input_size, hidden_size)
        self.attn_weight = MLP(input_size=3 * hidden_size, hidden_sizes=[attention_size, 1], dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        max_len = x.shape[1]
        mask = torch.tril(
                    torch.ones((1, max_len, max_len), dtype=torch.bool, device=x.device)
                )  # (1, L, L)
        x = self.ln(x)  # (B, L, H)
        x1 = x.unsqueeze(1).repeat_interleave(max_len, 1)
        x2 = x.unsqueeze(2).repeat_interleave(max_len, 2)
        attn_weight = self.attn_weight(torch.cat((x1, x2, x1 * x2), -1)).squeeze(-1)  # (B, L, L)
        attn_weight *= mask
        return torch.matmul(attn_weight, x)