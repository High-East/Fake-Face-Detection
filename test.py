'''
USAGE:
python test.py --config_file configs/ResNet.yaml 
'''

import os
import random
from psutil import virtual_memory
import fire
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2

from flags import Flags
from utils import get_network, guarantee_numpy
from checkpoint import load_checkpoint
from dataset import FaceDataset
from metrics import accuracy, precision, recall
from cam import apply_cam


def main(config_file, cam=False):
    options = Flags(config_file).get()

    # Set random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    os.environ["PYTHONHASHSEED"] = str(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    is_cuda = torch.cuda.is_available()
    print("--------------------------------")
    print("Running {} on device {}\nWARNING: THIS IS TEST MODE!!\n".format(options.network, options.device))

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    torch.cuda.empty_cache()
    print(
        "[+] System environments\n",
        "Device: {}\n".format(torch.cuda.get_device_name(current_device)),
        "Random seed : {}\n".format(options.seed),
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    model = get_network(options)

    checkpoint = load_checkpoint(options.test_checkpoint, cuda=is_cuda)
    model.load_state_dict(checkpoint['model'])
    model.to(options.device)
    model.eval()

    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Checkpoint: {}\n".format(options.test_checkpoint),
        "Model parameters: {:,}\n".format(
            sum(p.numel() for p in model.parameters()),
        ),
    )

    summary(model, (3, 224, 224), 32)

    w = options.input_size.width
    h = options.input_size.height

    transforms_test = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    test = pd.read_csv(options.data.test)
    test['path'] = test['path'].map(lambda x: './data' + x[12:])

    test_dataset = FaceDataset(image_label=test, transforms=transforms_test)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=options.data.random_split,
                                 num_workers=options.num_workers)

    losses = []
    acces = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(test_dataloader), leave=True):
            images = images.to(options.device, torch.float)
            targets = targets.to(options.device, torch.long)

            scores = model(images).to(options.device)
            _, preds = scores.max(dim=1)

            loss = F.cross_entropy(scores, targets)
            acc = accuracy(targets, preds, options.batch_size)
            pre = precision(targets, preds)
            rec = recall(targets, preds)

            losses.append(loss.item())
            acces.append(acc)
            precisions.append(pre)
            recalls.append(rec)

    print(
        "[+] Test result\n",
        "{:10s}: {:2.8f}\n".format('Loss', np.mean(losses)),
        "{:10s}: {:2.8f}\n".format('Accuracy', np.mean(acces)),
        "{:10s}: {:2.8f}\n".format('Precision', np.mean(precisions)),
        "{:10s}: {:2.8f}\n".format('Recall', np.mean(recalls)),
    )

    # GradCAM
    if cam:
        for idx in range(16):
            rgb_img = images[idx].permute(1, 2, 0)
            if targets[idx] == 0:
                label = 'fake'
            else:
                label = 'real'
            if options.network == 'ResNet':
                target_layers = [model.layer1[-1], model.layer2[-1], model.layer3[-1], model.layer4[-1]]
                for i, target in enumerate(target_layers, 1):
                    apply_cam(model, guarantee_numpy(rgb_img), [target],
                              title=f"{options.network} {i}th layer [{label}]")
            elif options.network == 'EfficientNet':
                target_layers = [model._blocks[-1]]
                for i, target in enumerate(target_layers, 16):
                    apply_cam(model, guarantee_numpy(rgb_img), [target],
                              title=f"{options.network} {i}th layer [{label}]")
            else:
                raise ValueError(f"Not supported {options.network} for CAM.")


if __name__ == '__main__':
    fire.Fire(main)
