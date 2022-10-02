"""Modules of ImageTransformer.
"""
from importlib.resources import path
from math import prod
from black import out
import torch
from torch import nn
import torch.nn.functional as F

from ..config.config import Config


class ImageToSequence(nn.Module):
    """Transforms image into sequence for self-attention module.

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes ImageToSequence module."""
        super().__init__()

        cfg_attention = config.transformer.self_attention
        self.sequence_length = cfg_attention.sequence_length
        self.embedding_dim = cfg_attention.n_heads * cfg_attention.head_dim

        img_channels, img_height, img_width = config.data.input_shape

        out_channels = config.transformer.image_to_sequence.out_channels
        patch_size = config.transformer.image_to_sequence.patch_size

        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        in_features = out_channels * (img_height // patch_size) * (img_width // patch_size)
        out_features = self.sequence_length * self.embedding_dim
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        We don't need an extra embedding or positional encoding when  we
        work with image data.

        Images are of size (batch_size, num_channels * width * height) and
        are transformed to size (batch_size, sequence_size * embedding_dim)

        Returns:
            Linear transformation.
        """
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, self.sequence_length, self.embedding_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention for image data."""

    def __init__(self, config: Config) -> None:
        """Initializes multi-head self-attention module."""
        super().__init__()

        cfg = config.transformer.self_attention

        self.sequence_length = cfg.sequence_length
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.embedding_dim = cfg.n_heads * cfg.head_dim
        self.dropout_prob = cfg.dropout_prob
        self.use_bias = cfg.use_bias

        bias = True if self.use_bias else False

        self.comp_keys = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=bias
        )
        self.comp_queries = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=bias
        )
        self.comp_values = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=bias
        )
        self.linear = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=bias
        )

        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        batch_size = x.shape[0]

        # Compute keys, queries, and values over all embedding vectors.
        keys = self.comp_keys(x)
        queries = self.comp_queries(x)
        values = self.comp_values(x)

        # Split keys, queries, and values for processing in different heads.
        keys = keys.view(-1, self.sequence_length, self.n_heads, self.head_dim)
        queries = queries.view(-1, self.sequence_length, self.n_heads, self.head_dim)
        values = values.view(-1, self.sequence_length, self.n_heads, self.head_dim)

        # Prepare keys, queries, and values for batch matrix mulitplication.
        keys = keys.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_length, self.head_dim
        )
        queries = queries.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_length, self.head_dim
        )
        values = values.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_length, self.head_dim
        )

        # First part of scaled dot-product self-attention
        # Compute attention weights
        out = torch.bmm(queries, keys.transpose(1, 2)) / self.embedding_dim**0.5
        out = F.softmax(out, dim=2)
        out = self.dropout(out)

        # Second part of scaled dot-product self-attention.
        out = torch.bmm(out, values)

        # Return back to original shape.
        out = out.view(-1, self.n_heads, self.sequence_length, self.head_dim)
        out = out.transpose(1, 2).reshape(
            -1, self.sequence_length, self.n_heads * self.head_dim
        )

        # Unify all heads in linear transformation.
        out = self.linear(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Module consisting of self-attention and full connected layers."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        cfg_block = config.transformer.transformer_block
        hidden_expansion = cfg_block.hidden_expansion
        dropout_prob = cfg_block.dropout_prob

        cfg_attention = config.transformer.self_attention
        embedding_dim = cfg_attention.n_heads * cfg_attention.head_dim

        self.attention = MultiHeadSelfAttention(config)

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(hidden_expansion * embedding_dim)),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(int(hidden_expansion * embedding_dim), embedding_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class Classifier(nn.Module):
    """Transforms attention head outputs to class predictions."""

    def __init__(self, config: Config) -> None:
        """Initializes the classifier."""
        super().__init__()
        cfg_attention = config.transformer.self_attention
        sequence_length = cfg_attention.sequence_length
        embedding_dim = cfg_attention.n_heads * cfg_attention.head_dim
        n_dims_out = config.data.n_classes

        self.linear = nn.Linear(
            in_features=sequence_length * embedding_dim,
            out_features=n_dims_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)
        return out
