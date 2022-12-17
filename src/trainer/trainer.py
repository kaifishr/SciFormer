import datetime
import os
import time

import torch
from torch import nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from ..utils.stats import comp_stats_classification
from ..summary.summary import (
    add_graph,
    add_input_samples,
    add_hist_params,
    add_hparams,
    add_patch_embedding_weights,
    add_token_embedding_weights,
    add_position_embedding_weights,
    add_mask_weights,
)
from ..config.config import Config


def trainer(model: torch.nn.Module, dataloader: tuple, config: Config) -> None:
    """

    Args:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Class holding configuration.

    """
    runs_dir = config.dirs.runs
    dataset = config.dataloader.dataset
    uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    tag = config.tag

    log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

    writer = SummaryWriter(log_dir=log_dir)
    run_training(model=model, dataloader=dataloader, writer=writer, config=config)
    writer.close()


def run_training(model, dataloader, writer, config: Config) -> None:
    """Main training logic.

    Trains passed model with data coming from dataloader.

    Args:
        model: PyTorch model.
        dataloader: Training and test data loader.
        writer: Tensorboard writer instance.
        config: Class holding configuration.

    """
    tag = config.tag
    device = config.trainer.device
    dataset = config.dataloader.dataset
    num_update_steps = config.trainer.num_update_steps
    step_size = config.trainer.lr_step_size
    gamma = config.trainer.lr_gamma

    train_loader, test_loader = dataloader

    # Add graph of model to Tensorboard.
    if config.summary.add_graph:
        add_graph(model=model, dataloader=train_loader, writer=writer, config=config)

    # Add sample batch to Tensorboard.
    if config.summary.add_sample_batch:
        add_input_samples(
            dataloader=train_loader, writer=writer, tag="train", global_step=0
        )
        add_input_samples(
            dataloader=test_loader, writer=writer, tag="test", global_step=0
        )

    learning_rate = config.trainer.learning_rate
    weight_decay = config.trainer.weight_decay
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    update_step = 0  # num_updates?

    while update_step < num_update_steps:

        running_loss = 0.0
        running_accuracy = 0.0
        running_counter = 0

        model.train()
        t0 = time.time()

        for x_data, y_data in train_loader:

            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = x_data.to(device), y_data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Feedforward
            outputs = model(inputs)

            # Reshape outputs and labels as if we have a classification task.
            # TODO: Move reshaping to model.
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)

            # Gradient descent
            optimizer.step()
            update_step += 1

            # keeping track of statistics
            running_loss += loss.item()
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
            running_counter += labels.size(0)

            print(update_step)
            if config.summary.save_train_stats.every_n_updates > 0:
                if update_step % config.summary.save_train_stats.every_n_updates == 0:
                    train_loss = running_loss / running_counter
                    train_accuracy = running_accuracy / running_counter

                    writer.add_scalar("train_loss", train_loss, global_step=update_step)
                    writer.add_scalar("train_accuracy", train_accuracy, global_step=update_step)

                    time_per_update = (time.time() - t0) / update_step 
                    writer.add_scalar("time_per_update", time_per_update, global_step=update_step)
                    writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], global_step=update_step)

                    running_loss = 0.0
                    running_accuracy = 0.0
                    running_counter = 0

                    t0 = time.time()
                    scheduler.step()

                    print(f"{update_step:09d} {train_loss:.5f} {train_accuracy:.4f}")

            if config.summary.save_test_stats.every_n_updates > 0:
                if update_step % config.summary.save_test_stats.every_n_epochs == 0:
                    test_loss, test_accuracy = comp_stats_classification(
                        model=model,
                        criterion=criterion,
                        data_loader=test_loader,
                        device=device,
                    )
                    writer.add_scalar("test_loss", test_loss, global_step=update_step)
                    writer.add_scalar("test_accuracy", test_accuracy, global_step=update_step)

                    if config.summary.add_hparams:
                        add_hparams(
                            writer,
                            config,
                            train_loss,
                            train_accuracy,
                            test_loss,
                            test_accuracy,
                        )

            if config.summary.save_model.every_n_updates > 0:
                if update_step % config.summary.save_model.every_n_updates == 0:
                    model_name = f"{dataset}{f'_{tag}' if tag else ''}.pth"
                    model_path = os.path.join(config.dirs.weights, model_name)
                    torch.save(model.state_dict(), model_path)

            if config.summary.add_params_hist.every_n_updates > 0:
                if update_step % config.summary.add_params_hist.every_n_updates == 0:
                    add_hist_params(model=model, writer=writer, global_step=update_step)

            if config.summary.add_patch_embeddings.every_n_updates > 0:
                if update_step % config.summary.add_patch_embeddings.every_n_updates == 0:
                    add_patch_embedding_weights(model=model, writer=writer, global_step=update_step)

            if config.summary.add_token_embeddings.every_n_updates > 0:
                if update_step % config.summary.add_token_embeddings.every_n_updates == 0:
                    add_token_embedding_weights(model=model, writer=writer, global_step=update_step)

            if config.summary.add_position_embeddings.every_n_updates > 0:
                if update_step % config.summary.add_position_embeddings.every_n_updates == 0:
                    add_position_embedding_weights(model=model, writer=writer, global_step=update_step)

            if config.summary.add_mask_weights.every_n_updates > 0:
                if update_step % config.summary.add_mask_weights.every_n_updates == 0:
                    add_mask_weights(model=model, writer=writer, global_step=update_step)