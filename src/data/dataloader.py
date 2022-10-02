import os
import tarfile

import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url

from ..config.config import Config


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: Config) -> tuple[DataLoader, DataLoader]:
    """Creates dataloader for specified dataset."""

    dataset = config.dataloader.dataset 
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size

    if dataset == "imagewoof":

        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
        download_url(url=dataset_url, root="./data")

        cwd = os.getcwd()
        with tarfile.open(cwd + "/data/imagewoof-160.tgz", "r:gz") as tar:
            tar.extractall(path=cwd + "/data")

        avg = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        stats = (avg, std)

        train_transform = transforms.Compose(
            [
                transforms.Resize(size=160),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, hue=0.3),
                transforms.RandomRotation(degrees=(0, 45)),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(size=(128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ]
        )

        data_dir = cwd + "/data/imagewoof-160/"
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + "/train", transform=train_transform
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + "/val", transform=test_transform
        )

        # Add number of classes and input shape to config
        config.data.n_classes = 10
        config.data.input_shape = (3, 128, 128)

    elif dataset == "cifar10":

        cifar10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        mean = np.mean(
            np.array(cifar10.data / 255.0), axis=(0, 1, 2)
        )  # [0.49139968 0.48215841 0.44653091]
        std = np.std(
            np.array(cifar10.data / 255.0), axis=(0, 1, 2)
        )  # [0.24703223 0.24348513 0.26158784]

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

        # Add number of classes and input shape to config
        config.data.n_classes = 10
        config.data.input_shape = (3, 32, 32)

    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    if "cuda" in str(config.trainer.device):
        pin_memory = True
    else:
        pin_memory = False

    trainloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        pin_memory=pin_memory,
    )

    testloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return trainloader, testloader
