import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from icecream import ic
from PIL import Image

from submodules.pytorch_grad_cam.grad_cam import GradCAM
from submodules.pytorch_grad_cam.ablation_cam import AblationCAM
from submodules.pytorch_grad_cam.xgrad_cam import XGradCAM
from submodules.pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from submodules.pytorch_grad_cam.score_cam import ScoreCAM
from submodules.pytorch_grad_cam.eigen_cam import EigenCAM
from submodules.pytorch_grad_cam.eigen_grad_cam import EigenGradCAM
from submodules.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel

from submodules.pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from submodules.pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)

from torchvision import transforms


def run_cam(
    model,
    target_layer,
    target_category,
    img_path,
    method,
    aug_smooth=False,
    eigen_smooth=False,
    use_cuda=True,
):
    use_cuda = ic(use_cuda and torch.cuda.is_available())

    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
    }

    if method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    cam = methods[method](model=model, target_layer=target_layer, use_cuda=use_cuda)

    # rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255

    rgb_img = Image.open(img_path)

    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        target_category=target_category,
        aug_smooth=True,
        eigen_smooth=False,
    )

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    # The image needs to be normalized before adding the mask.
    normalized_rgb_img = np.float32(rgb_img)/255
    normalized_rgb_img = transforms.Resize(224)(rgb_img)
    normalized_rgb_img = np.float32(normalized_rgb_img)/255

    # overlay cam image onto the original image.
    results = show_cam_on_image(normalized_rgb_img, grayscale_cam, use_rgb=True)

    return results

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    # gb = gb_model(input_tensor, target_category=target_category)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # results["cam_mask"] = cam_mask
    # results["cam_gb"] = cam_gb
    # results["gb"] = gb
