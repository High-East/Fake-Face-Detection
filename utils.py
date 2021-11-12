import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm


def get_network(
        options
):
    model = None

    if options.network == "ResNet":
        model = models.resnet18(pretrained=options.model.pretrained)
        model.fc = nn.Linear(512, options.data.num_classes)

    else:
        raise NotImplementedError

    return model.to(options.device)


def get_optimizer(
        params,
        options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params, lr=options.optimizer.lr, weight_decay=options.optimizer.weight_decay)
    else:
        raise NotImplementedError

    return optimizer


def guarantee_numpy(data):
    data_type = type(data)
    if data_type == torch.Tensor:
        device = data.device.type
        if device == 'cpu':
            data = data.detach().numpy()
        else:
            data = data.detach().cpu().numpy()
        return data
    elif data_type == np.ndarray or data_type == list:
        return data
    else:
        raise ValueError("Check your data type.")
