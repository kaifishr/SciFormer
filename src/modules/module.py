"""Modules of ImageTransformer.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.config.config import Config


class ImageToSequence(nn.Module):
    """Transforms image into sequence.

    Performs an embedding of images into a sequence.

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

        patch_size = config.transformer.image_to_sequence.patch_size

        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.sequence_length,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        self.linear = nn.Linear(
            in_features=(img_height // patch_size) * (img_width // patch_size),
            out_features=self.embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method.

        We don't need an extra embedding or positional encoding when  we
        work with image data.

        Images are of size (batch_size, num_channels * width * height) and
        are transformed to size (batch_size, sequence_size * embedding_dim)

        Returns:
            Tensor representing sequence of tokens.
        """
        x = self.conv(x)
        x = torch.flatten(input=x, start_dim=2, end_dim=-1)
        x = self.linear(x)
        x = x.view(-1, self.sequence_length, self.embedding_dim)
        return x


class TokenEmbedding(nn.Module):
    """Token embedding module.

    Embeds an integer as a vector of defined dimension.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens

        n_heads = config.transformer.self_attention.n_heads
        head_dim = config.transformer.self_attention.head_dim
        embedding_dim = n_heads * head_dim

        self.cfg_token_embedding = config.transformer.token_embedding
        size = (num_tokens, embedding_dim)

        if self.cfg_token_embedding.encoding == "random_normal":
            embedding = torch.normal(mean=0.0, std=0.01, size=size)
        elif self.cfg_token_embedding.encoding == "sinusoidal":
            embedding = self._sinusoidal_encoding(size=size)
        else:
            raise NotImplementedError(
                f"Embedding {self.cfg_token_embedding.encoding} not implemented."
            )

        requires_grad = True if self.cfg_token_embedding.is_trainable else False
        self.embedding = nn.Parameter(data=embedding, requires_grad=requires_grad)

        # TODO: Use PyTorch embedding and assign desired weight?
        # self.embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)

    @staticmethod
    def _sinusoidal_encoding(size: tuple) -> torch.Tensor:
        """Sinusoidal encoding scheme.

        See also: https://arxiv.org/abs/1706.03762
        """
        num_tokens, embedding_dim = size
        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(num_tokens, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Receives sequences of token identifiers and returns embedding.

        Args:
            x: Integer tensor holding integer token identifiers.

        Returns:
            Embedded tokens.
        """
        # x = self.embedding(x)  # TODO: use this later with nn.Embedding
        x = self.embedding[x]  # TODO: Test. Seems to work as well.
        # x = F.embedding(x, self.embedding)  # TODO: Test.
        return x


class PositionEmbedding(nn.Module):
    """Positional embedding module.

    Positional embedding with different encoding schemes.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()
        max_sequence_length = config.transformer.max_sequence_length
        n_heads = config.transformer.self_attention.n_heads
        head_dim = config.transformer.self_attention.head_dim
        embedding_dim = n_heads * head_dim

        self.pos_emb = config.transformer.position_embedding

        if self.pos_emb.is_activated:
            requires_grad = True if self.pos_emb.is_trainable else False
            size = (max_sequence_length, embedding_dim)

            if self.pos_emb.encoding == "zeros":
                embedding = torch.zeros(size=size)
            elif self.pos_emb.encoding == "ones":
                embedding = torch.ones(size=size)
            elif self.pos_emb.encoding == "random_normal":
                embedding = torch.normal(mean=0.0, std=0.01, size=size)
            elif self.pos_emb.encoding == "sinusoidal":
                embedding = self._sinusoidal_encoding(size=size)
            else:
                raise NotImplementedError(
                    f"Embedding {self.pos_emb.encoding} not implemented."
                )

            self.embedding = nn.Parameter(data=embedding, requires_grad=requires_grad)

    @staticmethod
    def _sinusoidal_encoding(size: tuple) -> torch.Tensor:
        """Sinusoidal encoding scheme.

        See also: https://arxiv.org/abs/1706.03762
        """
        max_sequence_length, embedding_dim = size
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_emb.is_activated:
            # pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)#.unsqueeze(0)
            # print(f"{self.embedding[pos].shape = }")
            # print(f"{self.embedding[:x.size(1)].shape = }")
            sequence_length = x.size(1)
            x = x + self.embedding[:sequence_length]
        return x


class Mask(nn.Module):
    """Implements a Mask module.

    Comes with different types of mask applied to the attention matrix
    of dot-products. The masks weight can be trained if required.
    """

    def __init__(self, config: Config):
        """Initializes the Mask module."""
        super().__init__()

        self.cfg_mask = config.transformer.mask

        if self.cfg_mask.is_activated:
            self.max_sequence_length = config.transformer.max_sequence_length
            size = (self.max_sequence_length, self.max_sequence_length)

            # self.mask = self._install_mask(config)
            mask_type = self.cfg_mask.type

            # Create masks.
            if mask_type == "trainable_additive":
                self.mask = nn.Parameter(
                    data=torch.zeros(size=size), requires_grad=True
                )
            elif mask_type == "trainable_multiplicative":
                self.mask = nn.Parameter(data=torch.ones(size=size), requires_grad=True)
            elif mask_type == "causal":
                self.mask = nn.Parameter(
                    data=torch.tril(input=torch.ones(size=size)),
                    requires_grad=False,
                )
            else:
                raise NotImplementedError(f"Mask {mask_type} not implemented.")
            # TODO: self.mask -> self.weight?

            self.mask_function = None
            self._install_mask(mask_type)

    def _install_mask(self, mask_type: str) -> None:
        # Create masks.
        if mask_type == "trainable_additive":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] + x
        elif mask_type == "trainable_multiplicative":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] * x
        elif mask_type == "causal":
            self.mask_function = lambda x, seq_len: x.masked_fill(
                self.mask[:seq_len, :seq_len] == 0, float("-inf")
            )
        else:
            raise NotImplementedError(f"Mask {mask_type} not implemented.")

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Applies installed mask to input tensor.

        Args:
            x: Input tensor.

        Returns: Masked tensor.
        """
        sequence_length = x.size(-1)
        x = self.mask_function(x, sequence_length)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg_mask.is_activated:
            x = self._apply_mask(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention for image data."""

    def __init__(self, config: Config) -> None:
        """Initializes multi-head self-attention module."""
        super().__init__()

        cfg = config.transformer.self_attention

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout_prob = cfg.dropout_prob
        self.use_bias = cfg.use_bias
        # self.use_mask = cfg.use_mask

        embedding_dim = cfg.n_heads * cfg.head_dim
        bias = True if self.use_bias else False

        self.comp_keys = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )
        self.comp_queries = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )
        self.comp_values = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )

        # Trainable mask. Let the network decide how the mask should look like.
        self.mask = Mask(config=config)

        self.linear = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )

        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        batch_size, sequence_length, embedding_dim = x.size()

        # Compute keys, queries, and values over all embedding vectors.
        keys = self.comp_keys(x)
        queries = self.comp_queries(x)
        values = self.comp_values(x)

        # Split keys, queries, and values for processing in different heads.
        keys = keys.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        values = values.view(batch_size, sequence_length, self.n_heads, self.head_dim)

        # Scaled dot-product self-attention
        out = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) / embedding_dim**0.5

        out = self.mask(out)
        # if self.use_mask:
        #     # out = self.mask + out
        #     # out = self.mask * out
        #     out.masked_fill_(self.mask == 0, float("-inf"))

        out = F.softmax(out, dim=-1)
        out = self.dropout(out)

        # Second part of scaled dot-product self-attention.
        out = torch.einsum("bhql,blhd->bqhd", [out, values])
        out = out.reshape(batch_size, sequence_length, self.n_heads * self.head_dim)

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
        sequence_length = (
            cfg_attention.sequence_length
        )  # TODO: For image transformer use here "config.transformer.img_to_sequence.sequence_length"
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
