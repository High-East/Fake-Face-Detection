import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import wandb


def get_network(
        options
):
    if options.network == "ResNet":
        model = models.resnet18(pretrained=options.model.pretrained)
        model.fc = nn.Linear(512, options.data.num_classes)

    elif options.network == "EfficientNet":
        if options.model.pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            model = EfficientNet.from_name('efficientnet-b0')

        model._fc = nn.Linear(1280, options.data.num_classes)

    elif options.network == "DenseNet":
        model = models.densenet121(pretrained=options.model.pretrained)
        model.classifier = nn.Linear(1024, options.data.num_classes)

    elif options.network == "VGGNet":
        model = models.vgg16(pretrained=options.model.pretrained)
        model.classifier[6] = nn.Linear(4096, options.data.num_classes)

    else:
        raise NotImplementedError

    return model.to(options.device)


def get_optimizer(
        params,
        options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params,
                               lr=float(options.optimizer.lr),
                               weight_decay=float(options.optimizer.weight_decay))
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


def read_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def initialize_wandb(config_path, options):
    config = read_yaml(config_path)
    if config['use_wandb']:
        wandb.login(key=config['key'])
        run = wandb.init(
            project=config['project'],
            config=options,
            notes=config['notes']
        )

        return run


class AttrDict(dict):
    def __init__(self, *config, **kwconfig):
        super(AttrDict, self).__init__(*config, **kwconfig)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        return None

    def get_values(self, keys):
        return {key: self.get(key) for key in keys}
