"""Main script to run experiments."""

from src.config.config import init_config
from src.data.dataloader import get_dataloader
from src.modules.model import (
    ImageTransformer, 
    CharacterTransformer
)
from src.trainer.trainer import Trainer
from src.utils.tools import (
    set_random_seed, 
    load_checkpoint, 
    count_model_parameters
)


def experiment_image():

    # Get configuration file
    config = init_config(file_path="config.yml")

    # Seed random number generator
    set_random_seed(seed=config.random_seed)

    # Get dataloader
    dataloader = get_dataloader(config=config)

    # Get the model
    model = ImageTransformer(config=config)
    count_model_parameters(model=model)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


def experiment_text():

    # Get configuration file.
    config = init_config(file_path="config.yml")

    # Seed random number generator.
    set_random_seed(seed=config.random_seed)

    # Get dataloader.
    dataloader = get_dataloader(config=config)

    # Get the model.
    model = CharacterTransformer(config=config)
    count_model_parameters(model=model)

    # Load pre-trained model.
    if config.load_model.is_activated:
        load_checkpoint(
            model=model,
            ckpt_dir=config.dirs.weights,
            model_name=config.load_model.model_name,
        )

    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


def main():
    experiment_text()


if __name__ == "__main__":
    main()
