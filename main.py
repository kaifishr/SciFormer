import json

from src.modules.model import ImageTransformer
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.trainer.trainer import trainer
from src.utils.tools import set_random_seed


def experiment_cifar10():

    # Get configuration file
    config = init_config(file_path="config.yml")

    # Seed random number generator
    set_random_seed(seed=config.random_seed)

    # Get dataloader
    dataloader = get_dataloader(config=config)

    # Get the model
    model = ImageTransformer(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    experiment_cifar10()


if __name__ == "__main__":
    main()
