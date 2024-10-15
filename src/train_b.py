import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_one_epoch(train_dataloader, model, optimizer, loss, use_mps=False):
    """
    Performs one training epoch
    """

    # Select device based on MPS availability
    device = torch.device("mps") if use_mps and torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # Set the model to training mode
    model.train()
    
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to the selected device (MPS or CPU)
        data, target = data.to(device), target.to(device)

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # Calculate the loss
        loss_value = loss(output, target)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Update average training loss
        train_loss += (1 / (batch_idx + 1)) * (loss_value.item() - train_loss)

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss, use_mps=False):
    """
    Validate at the end of one epoch
    """

    device = torch.device("mps") if use_mps and torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # Move data to the selected device (MPS or CPU)
            data, target = data.to(device), target.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # Calculate the loss
            loss_value = loss(output, target)

            # Calculate average validation loss
            valid_loss += (1 / (batch_idx + 1)) * (loss_value.item() - valid_loss)

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False, use_mps=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        from livelossplot import PlotLosses
        from livelossplot.outputs import MatplotlibPlot

        liveloss = PlotLosses(outputs=[MatplotlibPlot()])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: reduce learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss, use_mps
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss, use_mps)

        # Print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # Save the model if validation loss has decreased
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # Update learning rate using the scheduler
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss, use_mps=False):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    device = torch.device("mps") if use_mps and torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # set the module to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # Move data to the selected device (MPS or CPU)
            data, target = data.to(device), target.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)

            # Calculate the loss
            loss_value = loss(logits, target)

            # Update average test loss
            test_loss += ((1 / (batch_idx + 1)) * (loss_value.item() - test_loss))

            # Convert logits to predicted class
            pred = logits.argmax(dim=1, keepdim=True)

            # Compare predictions to true label
            correct += torch.sum(pred.eq(target.view_as(pred))).item()
            total += target.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss

    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
