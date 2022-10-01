import json

from src.modules.model import ImageTransformer
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed


def experiment_cifar10():

    # Get configuration file
    config = init_config(file_path="config.yml")
    config["data"]["dataset"] = "cifar10"

    # Seed random number generator
    set_random_seed(seed=config["experiment"]["random_seed"])

    # Get dataloader
    dataloader = get_dataloader(config=config)
    print(json.dumps(config, indent=4))

    # Get the model
    model = ImageTransformer(config=config)
    model.to(config["device"])

    train(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    experiment_cifar10()


if __name__ == "__main__":
    main()