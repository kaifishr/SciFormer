"""Provides datasets for image classification and autoregressive text generation.
"""
import torch
from torch.utils.data import Dataset

from src.config.config import Config


class CharDataset(Dataset):
    """Character-level dataset.

    Generates encoded batches of character sequences.

    Attributes:
        data:
        config:
        char_to_index:
        index_to_char:
        num_chars:
    """

    def __init__(self, data: str, config: Config):

        self.data = data
        self.config = config

        self.max_sequence_length = config.transformer.max_sequence_length

        # TODO: Do this only once in a prepocessing step.
        chars = sorted(list(set(data)))

        # Create lookup-tables with character-index-pairs in both directions.
        self.char_to_index = {char: i for i, char in enumerate(chars)}
        self.index_to_char = {i: char for i, char in enumerate(chars)}

        self.num_tokens = len(chars)

        print(f"Number of characters: {len(data)/1e6:.3f} M\n")
        print(f"Unique characteres: {self.num_tokens}\n")

        # self.look_back = 200

    def __len__(self):
        return len(self.data) - self.max_sequence_length

    def __getitem__(self, idx):
        """Extracts sequence of characters from data.

        For data holding a sequence of characters

        data = "The quick brown Fox jumps"

        idx=4 and block_size=8, the following block
        of characters are extracted from the data
        sequence

        char_block = "quick bro"

        which is being encoded as a list of integers:

        encoded_block = [9, 1, 4, 8, 2, 5, 3, 7, 6]
                         q  u  i  c  k " " b  r  o

        From this list, the following input and target
        is created:

        x = [9, 1, 4, 8, 2, 5, 3, 7]
        y = [1, 4, 8, 2, 5, 3, 7, 6]

        Args:
            idx: Index to access string stored in data.
        """
        # Try to find the start of a sentence.
        # if idx > self.look_back:
        #     idx_offset = self.data[idx-self.look_back:idx].rfind(".")
        #     if idx_offset > -1:
        #         idx -= (self.look_back - idx_offset - 2)  # -2 adjusts for period followed by blank space.

        char_sequence = self.data[idx : idx + self.max_sequence_length + 1]
        int_sequence = [self.char_to_index[char] for char in char_sequence]
        x = torch.tensor(data=int_sequence[:-1], dtype=torch.long)
        y = torch.tensor(data=int_sequence[1:], dtype=torch.long)
        return x, y
