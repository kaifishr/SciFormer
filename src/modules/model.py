"""Collection of custom neural networks.

"""
import torch
import torch.nn as nn

from .module import (
    ImageToSequence,
    TokenEmbedding,
    PositionEmbedding,
    TransformerBlock,
    Classifier,
)

from src.config.config import Config


class ImageTransformer(nn.Module):
    """Isotropic multi-head self-attention transformer neural network."""

    def __init__(self, config: dict):
        """Initializes image transformer."""
        super().__init__()

        self.image_to_sequence = ImageToSequence(config)
        self.position_embedding = PositionEmbedding(config)

        n_blocks = config.transformer.n_blocks
        blocks = [TransformerBlock(config) for _ in range(n_blocks)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.classifier = Classifier(config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for all modules of ImageTransformer."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_to_sequence(x)
        x = self.position_embedding(x)
        x = self.transformer_blocks(x)
        x = self.classifier(x)
        return x


class SwapAxes(nn.Module):
    def __init__(self, axis0: int, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)


class TokenClassifier(nn.Module):
    """Classifier for next token prediction."""

    def __init__(self, config: Config) -> None:
        """Initializes Classifier class."""
        super().__init__()

        cfg_attention = config.transformer.self_attention
        embedding_dim = cfg_attention.n_heads * cfg_attention.head_dim
        num_classes = config.data.num_tokens
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SequenceClassifier(nn.Module):
    """Classifier for next sequence prediction."""

    def __init__(self, config: Config) -> None:
        """Initializes Classifier class."""
        super().__init__()

        max_sequence_length = config.transformer.max_sequence_length
        out_sequence_length = config.transformer.out_sequence_length
        num_heads = config.transformer.self_attention.n_heads
        head_dim = config.transformer.self_attention.head_dim
        embedding_dim = num_heads * head_dim
        num_classes = config.data.num_tokens

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            SwapAxes(axis0=-2, axis1=-1),
            nn.Linear(
                in_features=max_sequence_length, out_features=out_sequence_length
            ),
            SwapAxes(axis0=-2, axis1=-1),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


class CharacterTransformer(nn.Module):
    """Character-level isotropic multi-head self-attention
    transformer neural network."""

    def __init__(self, config: Config):
        """Initializes image transformer."""
        super().__init__()

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        n_blocks = config.transformer.n_blocks
        blocks = [TransformerBlock(config) for _ in range(n_blocks)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.classifier = SequenceClassifier(config=config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights for all modules of ImageTransformer."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Assert that maximum sequence length is not exceeded.
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.transformer_blocks(x)
        x = self.classifier(x)
        return x
