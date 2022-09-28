"""Collection of custom neural networks.

"""
from imp import init_frozen
from math import prod
from turtle import forward

import torch
import torch.nn as nn

from .modules import DenseBlock


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention for image data."""

    def __init__(self, sequence_size, embedding_dim, n_heads, head_dim, ) -> None:
        """Initializes multi-head self-attention module."""
        super().__init__()
        self.sequence_size = sequence_size
        # self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = x.view(-1, self.sequence_size, self.n_heads, self.head_dim)
        return x


class ImageTransformer(nn.Module):
    """Isotropic multi-head self-attention transformer neural network."""

    def __init__(self, config: dict):
        super().__init__()

        self.input_shape = config["input_shape"]
        self.n_dims_in = prod(self.input_shape)
        self.n_dims_out = config["n_classes"]

        # Parameters for multi-head self-attention.
        self.n_heads = 4
        self.head_dim = 8 
        self.sequence_size = 128
        self.embedding_dim = self.n_heads * self.head_dim

        self.n_dims_hidden = 128
        self.n_blocks = 4

        self.image_to_sequence = self._image_to_sequence()
        self.attention = MultiHeadSelfAttention(
            sequence_size=self.sequence_size,
            embedding_dim=self.embedding_dim,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )

        self.classifier = self._classifier()

        # 1: Transform image to sequence.
        # 2: Do the multi-head self-attention thing.
        # 3: Transform attention head output to classification.

        self._weights_init()

    def _image_to_sequence(self) -> nn.Module:
        """Transforms image to sequence.

        We don't need an extra embedding or positional encoding as we work
        with single images.

        Images are of size (batch_size, num_channels * width * height) and
        are transformed to size (batch_size, sequence_size * embedding_dim)

        Returns:
            Linear transformation.

        """
        out_features = self.sequence_size * self.embedding_dim
        return nn.Linear(
            in_features=self.n_dims_in, out_features=out_features, bias=False
        )

    def _classifier(self):
        """Classifier transforms attention head outputs to class predictions.
        """
        return nn.Linear(in_features=self.sequence_size * self.embedding_dim, out_features=self.n_dims_out)

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, x):
        print(f"{x.shape = }")
        x = torch.flatten(x, start_dim=1)
        print(f"{x.shape = }")
        x = self.image_to_sequence(x)
        print(f"{x.shape = }")  # [256, 3, 32, 32]
        x = x.view(-1, self.sequence_size, self.embedding_dim)
        print(f"{x.shape = }")
        x = self.attention(x)
        print(f"{x.shape = }")  
        x = torch.flatten(x, start_dim=1)
        print(f"{x.shape = }")  
        x = self.classifier(x)
        print(f"{x.shape = }")  
        exit()
        return x
