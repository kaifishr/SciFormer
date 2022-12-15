"""Provides datasets for image classification and autoregressive text generation.
"""
import torch
from torch.utils.data import Dataset

from src.config.config import Config


class CharDataset(Dataset):
    """Character-level dataset.
    
    Generates batches of character sequences.

    Attributes:
        data:
        config:
        char_to_index:
        index_to_char:
        num_chars:
    """

    def __init__(self, data, config: Config):

        self.data = data
        self.config = config

        # TODO: Do this only once in a prepocessing step.
        chars = sorted(list(set(data)))

        # Create lookup-tables with character-index-pairs in both directions.
        self.char_to_index = {char: i for i, char in enumerate(chars)}
        self.index_to_char = {i: char for i, char in enumerate(chars)}

        print(f"Total size of dataset: {len(data)} characters.")
        print(f"Unique characteres: {len(chars)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Extracts sequence of characters from data.

        data = [a, b, c, d, e, f, g, h, i, j, k]  
        x = [b, c, d, e, f]  
        y = [c, d, e, f, g]  
        """

        data = ["a", "b"]
        x = torch.tensor(data=data, dtype=torch.long)
        y = torch.tensor(data=data, dtype=torch.long)

        return x, y
