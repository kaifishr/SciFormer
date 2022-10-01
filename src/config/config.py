"""Methods for handling of configuration file.
"""
import yaml
from pathlib import Path

import torch

from .cfg import Config


def init_config(file_path: str) -> dict:
    """Initializes configuration class for experiment.

    Args:
        file_path: File to configuration file.

    Returns:
        Config class.

    """

    # Load yaml file as dictionary.
    config = load_config(file_path=file_path)

    # Convert dictionary to configuration class.
    config = Config(d=config)

    Path(config.dirs.data).mkdir(parents=True, exist_ok=True)
    Path(config.dirs.runs).mkdir(parents=True, exist_ok=True)
    Path(config.dirs.weights).mkdir(parents=True, exist_ok=True)

    # Check for accelerator
    if config.trainer.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config.trainer.device=device
    else:
        config.trainer.device == torch.device("cpu")

    return config


def load_config(file_path: str) -> dict:
    """Loads configuration file.

    Args:
        file_path: Path to yaml file.

    Returns:
        Dictionary holding content of yaml file.

    """
    with open(file_path, "r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    return config
