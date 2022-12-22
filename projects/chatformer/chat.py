"""Methods to chat with pre-trained transformer network."""
import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.data.dataloader import get_dataloader
from src.config.config import Config, init_config
from src.modules.model import CharacterTransformer
from src.utils.tools import load_checkpoint


class Chat:
    """Chat class.

    Uses a autoregressive model to generate text provided a prompt.

    Attributes:
        model: An autoregressive model.
        dataset: Dataset model has been trained on.
        config: Configuration.
        valid_characters: Legal characters.

    """

    def __init__(self, model: torch.nn.Module, dataset: Dataset, config: Config):
        """Initializes chat class."""
        self.model = model
        self.dataset = dataset
        self.config = config

        self.valid_characters = list(self.dataset.char_to_index)

        self.device = self.config.trainer.device
        self.max_sequence_length = self.config.transformer.max_sequence_length

        # Maximum number of generated tokens.
        self.max_num_tokens = 200
        self.temperature = 0.6
        self.do_sample = False
        self.top_k = 10

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generates text from prompt.

        Args:
            input_text: Prompt text.

        Returns:
            Text generated by model.
        """
        # Encode input characters as integer using lookup table from dataloader.
        data = [self.dataset.char_to_index[char] for char in prompt]

        # Create input tensor from encoded characters.
        x = torch.tensor(data=data, dtype=torch.long)[None, ...].to(self.device)

        # Generate some tokens
        for _ in range(self.max_num_tokens):

            # Make sure that the sequence length is smaller than max sequence length.
            sequence = (
                x
                if x.size(-1) <= self.max_sequence_length
                else x[:, -self.max_sequence_length :]
            )

            # Feed sequence into model.
            logits = self.model(sequence)

            # Extract probabilities for last token.
            logits = logits[:, -1, :]

            # High temperature: make model more creative (text generation).
            # Low temperature: make model more confident (knowledge retrieval).
            logits = logits / self.temperature

            # Convert logits to probabilities.
            probabilities = F.softmax(input=logits, dim=-1)

            if self.do_sample:
                index_next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                # Take the most likely next token.
                _, index_next_token = torch.topk(probabilities, k=1, dim=-1)

            # Add index of most likely token to running sequence.
            x = torch.cat((x, index_next_token), dim=-1)

        # Remove prompt from sequence:
        x = x[:, len(prompt) :]

        output = "".join([self.dataset.index_to_char[int(index)] for index in x[0]])

        return output

    def _is_valid_prompt(self, prompt: str) -> bool:
        """Checks if input prompt contains any illegal characters."""
        for character in prompt:
            if character not in self.valid_characters:
                print(f"Character '{character}' was not part of the training data.")
                return False
        return True

    def run(self):
        """Runs chat."""
        is_running = True

        # Test model with some simple prompts.
        prompts = [
            "1 is one bigger than 0. 2 is one bigger than 1. 3 is one bigger than 2. What is the sum of 1 and 2?",
        ]
        for prompt in prompts:
            print(f"\n{prompt}\n")
            if self._is_valid_prompt(prompt=prompt):
                output = self._generate(prompt=prompt)
                print(f"\n{output}\n")

        while is_running:
            print("\nPlease enter a prompt.\n")

            prompt = input()

            if prompt == "exit":
                is_running = False
            elif prompt == "":
                continue

            # Feed text to model
            if is_running and self._is_valid_prompt(prompt=prompt):
                output = self._generate(prompt=prompt)
                print(f"\n{output}\n")

        print("Bye!")


if __name__ == "__main__":

    cwd = os.getcwd()
    print(cwd)

    # Get configuration file
    # config_path = "weights/config4.yml"
    config_path = "config.yml"
    config = init_config(file_path=config_path)

    # Get dataloader and dataset.
    # . Load dataloader to initialize config.
    # . Get dataset with encoder-decoder methods from dataloader.
    dataloader, _ = get_dataloader(config=config)
    dataset = dataloader.dataset

    # Get the model
    model = CharacterTransformer(config=config)
    # model = torch.jit.script(model)

    ckpt_dir = config.dirs.weights
    model_name = "lexicap"
    load_checkpoint(model=model, ckpt_dir=ckpt_dir, model_name=model_name)
    config.trainer.device = torch.device("cpu")
    model.to(config.trainer.device)
    model.eval()

    chat = Chat(model=model, dataset=dataset, config=config)
    chat.run()
