"""Random search module.

Creates random sets of hyperparameters based
on hparams.yml configuration file.

"""
import random
import time

from ..config.config import Config


def create_random_config_(config: Config) -> None:
    """Creates random confuguration."""

    random.seed(time.time())

    # learning_rate = 10 ** random.uniform(-5, -3)
    # config.trainer.learning_rate = learning_rate

    # batch_size = random.choice([16, 32, 64, 128])
    # config.trainer.batch_size = batch_size

    # n_blocks = random.choice([1, 2, 3, 4])
    # config.transformer.n_blocks = n_blocks

    patch_size = random.choice([1, 2, 4])
    config.transformer.image_to_sequence.patch_size = patch_size

    hidden_expansion = random.choice(range(1, 9))
    config.transformer.transformer_block.hidden_expansion = hidden_expansion

    dropout_prob = random.uniform(0.0, 0.2)
    config.transformer.transformer_block.dropout_prob = dropout_prob

    sequence_length = random.choice(range(1, 32))
    config.transformer.self_attention.sequence_length = sequence_length

    n_heads = random.choice(range(2, 16))
    config.transformer.self_attention.n_heads = n_heads

    head_dim = random.choice(range(2, 16))
    config.transformer.self_attention.head_dim = head_dim

    dropout_prob = random.uniform(0.0, 0.2)
    config.transformer.self_attention.dropout_prob = dropout_prob

    # use_bias = random.choice([True, False])
    # config.transformer.self_attention.use_bias = use_bias
