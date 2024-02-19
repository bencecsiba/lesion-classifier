import torch
import torch.nn as nn
import torch.optim

def get_loss():
    
    loss = nn.CrossEntropyLoss()

    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    
    if optimizer.lower() == "sgd":

        opt = torch.optim.SGD(
            
            model.parameters(),
            lr = learning_rate,
            momentum = momentum,
            weight_decay = weight_decay
        )

    elif optimizer.lower() == "adam":

        opt = torch.optim.Adam(
            
            model.parameters(),
            lr = learning_rate,
            
            weight_decay = weight_decay
            
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt