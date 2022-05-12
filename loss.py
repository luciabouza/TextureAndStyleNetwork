#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
from torch.nn.functional import mse_loss
import torch.nn.functional as F

# Computes Gram matrix for the input batch tensor.
#    Args: tnsr (torch.Tensor): input tensor of the Size([B, C, H, W]).
#    Returns:  G (torch.Tensor): output tensor of the Size([B, C, C]).
def gramm(tnsr: torch.Tensor) -> torch.Tensor: 

    b,c,h,w = tnsr.size() 
    F = tnsr.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1,2)) 
    #G.div_(h*w)
    G.div_(h*w*c) # Ulyanov
    return G

# Computes MSE Loss for 2 Gram matrices 
def gram_loss(input: torch.Tensor, gramm_target: torch.Tensor, weight: float = 1.0):
  
    loss = weight * mse_loss(gramm(input), gramm_target)
    return loss

# Computes MSE Loss for 2 tensors, with weight
def content_loss(input: torch.Tensor, target: torch.Tensor, weight: float = 1.0):
    
    loss = weight * mse_loss(input, target)
    return loss