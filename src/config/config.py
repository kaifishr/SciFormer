"""Methods for handling of configuration file.
"""
import yaml
from pathlib import Path

import torch

from .cfg import Config


def init_config(file_path: str) -> dict:
    """Initializes configuration for experiment.

    Args:
        file_path: File to configuration file.

    Returns:
        Dictionary holding configuration.

    """
    config = load_config(file_path=file_path)

    # Create directories
    for directory in config["dirs"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Check for accelerator
    if config["experiment"]["device"] == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)

    cfg = Config(d=config)
    print(cfg)
    exit()

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
