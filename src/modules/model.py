"""Collection of custom neural networks.

"""
import torch
import torch.nn as nn

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

        n_blocks = config.transformer.n_blocks

        cfg_attention = config.transformer.self_attention

        # self.position_embedding = nn.Embedding(
        #     num_embeddings=cfg_attention.sequence_length,
        #     embedding_dim=cfg_attention.n_heads * cfg_attention.head_dim,
        # )
        self.position_embedding = nn.Parameter(
            data=torch.normal(
                mean=0.0,
                std=0.02,
                size=(
                    cfg_attention.sequence_length,
                    cfg_attention.n_heads * cfg_attention.head_dim,
                ),
            ),
            requires_grad=True,
        )

        self.image_to_sequence = ImageToSequence(config)

        blocks = [TransformerBlock(config) for _ in range(n_blocks)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.classifier = Classifier(config)

        self._count_model_parameteres()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initiaializes weights for all modules of ImageTransformer."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _count_model_parameteres(self) -> None:
        """Computes number of model parameters."""
        n_params = [params.numel() for params in self.parameters()]
        print(f"{n_params = }")
        print(f"Number of parameters: {sum(n_params)/1e6:.2f} M")

    def forward(self, x):
        x = self.image_to_sequence(x)
        x = x + self.position_embedding  # TODO: check this
        x = self.transformer_blocks(x)
        x = self.classifier(x)
        return x
