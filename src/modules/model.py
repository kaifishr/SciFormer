"""Collection of custom neural networks.

"""
from imp import init_frozen
from math import prod
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import (
    ImageToSequence, 
    TransformerBlock, 
    Classifier,
)


class ImageTransformer(nn.Module):
    """Isotropic multi-head self-attention transformer neural network."""

    def __init__(self, config: dict):
        """Initializes image transformer."""
        super().__init__()

        self.input_shape = config["input_shape"]
        self.n_dims_in = prod(self.input_shape)
        self.n_dims_out = config["n_classes"]

        # Parameters for multi-head self-attention.
        self.n_heads = 8
        self.head_dim = 32
        self.sequence_size = 16
        self.embedding_dim = self.n_heads * self.head_dim
        self.n_blocks = 4

        self.image_to_sequence = ImageToSequence(
            n_dims_in=self.n_dims_in,
            sequence_size=self.sequence_size,
            embedding_dim=self.embedding_dim,
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    sequence_size=self.sequence_size,
                    embedding_dim=self.embedding_dim,
                    n_heads=self.n_heads,
                    head_dim=self.head_dim,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.classifier = Classifier(
            self.sequence_size,
            self.embedding_dim,
            self.n_dims_out,
        )

        self._count_model_parameteres()

        # Initialize model weights here. See mingpt.
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initiaializes weights for all modules of ImageTransformer."""


    def _count_model_parameteres(self) -> None:
        """Computes number of model parameters."""
        n_params = [params.numel() for params in self.parameters()]
        print(f"{n_params = }")
        print(f"Number of parameters: {sum(n_params)/1e6:.2f} M")

    def forward(self, x):
        x = self.image_to_sequence(x)
        x = self.transformer_blocks(x)
        x = self.classifier(x)
        return x
