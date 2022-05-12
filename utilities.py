#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Utilities
# Functions to manage images

import torch
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def prep_img(image: str, size=None, mean=MEAN, std=STD, both: bool = False):
    """Preprocess image.
    1) load as PIl
    2) resize
    3) convert to tensor
    5) remove alpha channel if any
    4) normalize
    """
    im = Image.open(image) #.convert('RGB')
    if both==True:
        texture = resize(im, (size,size))
    else: 
        texture = resize(im, size)
    texture_tensor = to_tensor(texture).unsqueeze(0)
    if texture_tensor.shape[1]==4:
        print('removing alpha chanel')
        texture_tensor = texture_tensor[:,:3,:,:]
    texture_tensor = normalize(texture_tensor, mean=mean, std=std)
    return texture_tensor


def denormalize(tensor: torch.Tensor, mean=MEAN, std=STD, inplace: bool = False):
    """Based on torchvision.transforms.functional.normalize.
    """
    tensor = tensor.clone().squeeze() 
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def to_pil(tensor: torch.Tensor):
    """Converts tensor to PIL Image.
    Args: tensor (torch.Temsor): input tensor to be converted to PIL Image of torch.Size([C, H, W]).
    Returns: PIL Image: converted img.
    """
    img = tensor.clone().detach().cpu()
    img = denormalize(img).clip(0, 1)
    img = to_pil_image(img)
    return img
