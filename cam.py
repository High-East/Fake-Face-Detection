import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (show_cam_on_image, preprocess_image)


def apply_cam(
        model,
        rgb_img: np.ndarray,  # (H, W, C)
        target_layers,
        method=GradCAM,
        use_cuda=True,
        title=None
):
    with method(model=model,
                target_layers=target_layers,
                use_cuda=use_cuda) as cam:
        input_tensor = torch.FloatTensor(rgb_img).permute(2, 0, 1).unsqueeze(0)
        cam.batch_size = 1

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=None,
                            aug_smooth=None,
                            eigen_smooth=None)  # (1, 224, 224)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]  # (224, 224)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        plt.imshow(cam_image)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()
