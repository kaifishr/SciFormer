"""Collection of custom blocks for neural networks.

"""
import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        linear_cfg = dict(in_features=in_features, out_features=out_features)

        self.linear1 = nn.Linear(**linear_cfg)
        self.linear2 = nn.Linear(**linear_cfg)

        self.bn1 = torch.nn.BatchNorm1d(num_features=out_features)
        self.bn2 = torch.nn.BatchNorm1d(num_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.linear1(x)
        out = torch.relu(out)
        out = self.bn1(out)

        out = self.linear2(out)
        out = torch.relu(out)
        out = self.bn2(out)

        return out + identity
