"""Main script to run experiments."""

from src.config.config import init_config
from src.data.dataloader import get_dataloader
from src.modules.model import ImageTransformer, CharacterTransformer
from src.random_search.random_search import create_random_config_
from src.trainer.trainer import trainer
from src.utils.tools import set_random_seed


def experiment_long_run():

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


def experiment_random_search():

    n_runs = 1000
    n_epochs = 10

    # Get configuration file
    config = init_config(file_path="config.yml")
    config.tag = "random_search"
    config.trainer.n_epochs = n_epochs

    for _ in range(n_runs):

        create_random_config_(config)
        print(config)

        # Seed random number generator
        set_random_seed(seed=config.random_seed)

        # Get dataloader
        dataloader = get_dataloader(config=config)

        # Get the model
        model = ImageTransformer(config=config)
        model.to(config.trainer.device)

        trainer(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def experiment_text():

    # Get configuration file
    config = init_config(file_path="config.yml")

    # Seed random number generator
    set_random_seed(seed=config.random_seed)

    # Get dataloader
    dataloader = get_dataloader(config=config)

    # Get the model
    model = CharacterTransformer(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    # experiment_long_run()
    # experiment_random_search()
    experiment_text()


if __name__ == "__main__":
    main()
