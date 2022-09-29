"""Modules of ImageTransformer.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ImageToSequence(nn.Module):
    """Transforms image into sequence for self-attention module.

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, n_dims_in: int, sequence_size: int, embedding_dim: int) -> None:
        """Initializes ImageToSequence module."""
        super().__init__()
        self.n_dims_in = n_dims_in
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim

        image_dim = 3
        image_height = 32
        image_width = 32

        in_channels = image_dim
        out_channels = 3
        patch_size = 2

        assert image_height % patch_size == 0

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        in_features = out_channels * (image_height // patch_size) * (image_width // patch_size)
        out_features = self.sequence_size * self.embedding_dim
        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

        self.layer_norm_1 = nn.LayerNorm(in_features)
        self.layer_norm_2 = nn.LayerNorm(out_features)

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
        x = self.layer_norm_1(x)
        x = self.linear(x)
        x = self.layer_norm_2(x)
        x = x.view(-1, self.sequence_size, self.embedding_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention for image data."""

    def __init__(
        self,
        sequence_size,
        embedding_dim,
        n_heads,
        head_dim,
    ) -> None:
        """Initializes multi-head self-attention module."""
        super().__init__()
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.comp_keys = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False
        )
        self.comp_queries = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False
        )
        self.comp_values = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False
        )

        self.linear = nn.Linear(
            in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        batch_size = x.shape[0]

        # Compute keys, queries, and values over all embedding vectors.
        keys = self.comp_keys(x)
        queries = self.comp_queries(x)
        values = self.comp_values(x)

        # Split keys, queries, and values for processing in different heads.
        keys = keys.view(-1, self.sequence_size, self.n_heads, self.head_dim)
        queries = queries.view(-1, self.sequence_size, self.n_heads, self.head_dim)
        values = values.view(-1, self.sequence_size, self.n_heads, self.head_dim)

        # Prepare keys, queries, and values for batch matrix mulitplication.
        keys = keys.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_size, self.head_dim
        )
        queries = queries.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_size, self.head_dim
        )
        values = values.transpose(1, 2).reshape(
            batch_size * self.n_heads, self.sequence_size, self.head_dim
        )

        # First part of scaled dot-product self-attention
        # Compute attention weights
        out = torch.bmm(queries, keys.transpose(1, 2)) / self.embedding_dim**0.5
        out = F.softmax(out, dim=2)

        # Second part of scaled dot-product self-attention.
        out = torch.bmm(out, values)

        # Return back to original shape.
        out = out.view(-1, self.n_heads, self.sequence_size, self.head_dim)
        out = out.transpose(1, 2).reshape(
            -1, self.sequence_size, self.n_heads * self.head_dim
        )

        # Unify all heads in linear transformation.
        out = self.linear(out)

        return out


class TransformerBlock(nn.Module):
    """Module consisting of self-attention and full connected layers."""

    def __init__(
        self,
        sequence_size: int,
        embedding_dim: int,
        n_heads: int,
        head_dim: int,
        hidden_multiplier: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.attention = MultiHeadSelfAttention(
            sequence_size=self.sequence_size,
            embedding_dim=self.embedding_dim,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )

        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_multiplier * self.embedding_dim),
            nn.ReLU(),
            nn.Linear(hidden_multiplier * self.embedding_dim, self.embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.
        """
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class Classifier(nn.Module):
    """Transforms attention head outputs to class predictions."""

    def __init__(self, sequence_size: int, embedding_dim: int, n_dims_out: int) -> None:
        """Initializes the classifier."""
        super().__init__()
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim
        self.n_dims_out = n_dims_out

        self.linear = nn.Linear(
            in_features=self.sequence_size * self.embedding_dim,
            out_features=self.n_dims_out,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)
        return out