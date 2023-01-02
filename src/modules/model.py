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
    SequenceClassifier,
    TokenClassifier
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

        # self.classifier = SequenceClassifier(config=config)
        self.classifier = TokenClassifier(config=config)

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
