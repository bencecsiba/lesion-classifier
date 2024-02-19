import tempfile
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(train_dataloader, model, optimizer, loss):

    if torch.backends.mps.is_available():
        model.to(torch.device("mps"))

    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):

        if torch.backends.mps.is_available():
            data, target = data.to(torch.device("mps")), target.to(torch.device("mps"))

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data)

        # 3. calculate the loss
        loss_value  = loss(output, target) 
        
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()
        
        # 5. perform a single optimization step (parameter update)
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss

def valid_one_epoch(valid_dataloader, model, loss):

    with torch.no_grad():

        model.eval()

        if torch.backends.mps.is_available():
            model.to(torch.device("mps"))

        valid_loss = 0.0
        correct = 0.
        total = 0.

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):

            if torch.backends.mps.is_available():
                data, target = data.to(torch.device("mps")), target.to(torch.device("mps"))

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data) 
            
            # 2. calculate the loss
            loss_value  = loss(output, target) # YOUR CODE HERE

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )
            
            preds  = output.data.max(1, keepdim=True)[1]
            correct += torch.sum(torch.squeeze(preds.eq(target.data.view_as(preds))).cpu())
            total += data.size(0)

        valid_accuracy = 100. * correct / total

    return valid_loss, valid_accuracy

def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    #if group_name.lower() == "loss":
        #ax.set_ylim([None, 4.5])


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    # Perform an initial validation of the model.
    valid_loss_min, _ = valid_one_epoch(data_loaders["valid"], model, loss) 
    valid_loss_min_count = 0

    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 

    for epoch in range(1, n_epochs + 1):
        logs = {} # Liveloss logs dictionary.
        
        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss, valid_accuracy = valid_one_epoch(data_loaders["valid"], model, loss)

        scheduler.step()

        # If the validation loss decreases by more than 1%, save the model
        if (valid_loss_min - valid_loss) / valid_loss_min > 0.01:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            # Test the model and log the test result.
            if interactive_tracking and (valid_loss_min_count % 3 == 0):
                _, logs["Test Accuracy"] = one_epoch_test(data_loaders['test'], model, loss)
                
            # Save the weights to save_path

            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["val_acc"] = valid_accuracy
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%".format(
                epoch, train_loss, valid_loss, valid_accuracy
            )
        )

def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    with torch.no_grad():

        model.eval()

        if torch.backends.mps.is_available():
            model.to(torch.device("mps"))

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):

            if torch.backends.mps.is_available():
                data, target = data.to(torch.device("mps")), target.to(torch.device("mps"))

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data) 
            
            # 2. calculate the loss
            loss_value  = loss(logits, target).detach() 

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            pred  = logits.data.max(1, keepdim = True)[1]

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    test_accuracy = 100. * correct / total
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        test_accuracy, correct, total))

    return test_loss, test_accuracy