#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from network import Pyramid2D
from utilities import *
from loss import *
from torchvision.transforms.functional import resize

# Identity function that normalizes the gradient on the call of backwards
# Used for "gradient normalization"
class Normalize_gradients(Function):
    @staticmethod
    def forward(self, input):
        return input.clone()
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.mul(1./torch.norm(grad_input, p=1))
        return grad_input


# Function to sample from network. 
def sample(generator, n_samples, sample_size, n_input_ch, folder, device):

    sz = [int(sample_size/i) for i in [1,2,4,8,16,32]]
    zk = [torch.rand(n_samples,n_input_ch,szk,szk) for szk in sz]
    z_samples = [Variable(z.to(device)) for z in zk ]

    for n in range(n_samples):
        batch_sample = generator(z_samples)
        sample = batch_sample[0,:,:,:].unsqueeze(0)
        out_img = to_pil(sample.squeeze(0))
        out_img.save(folder + '/offline_sample_' + str(n) + '.jpg', "JPEG")


def trainTextureSynthesis(device, input_name, max_iter = 5000, n_input_ch = 3, batch_size = 8, learning_rate = 0.001, lr_adjust = 750, lr_decay_coef = 0.66, use_GN = False):

    ######################################################################
    # create generator network

    gen = Pyramid2D(ch_in=3, ch_step=8)
    gen.to(device)

    ######################################################################
    # Load image

    # Prepare texture data
    input_image_name = './Textures/' + input_name
    img_size = 256

    target = prep_img(input_image_name, img_size).to(device)
    target_img = to_pil(target)

    if not os.path.exists('./Trained_models/' + input_name[:-4]):
        os.mkdir( './Trained_models/' + input_name[:-4])

    ######################################################################
    # Load VGG

    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    vgg.requires_grad_(False)

    # Initialize outputs dic
    outputs = {}

    # Hook definition
    def save_output(name):   
        # The hook signature
        def hook(module, module_in, module_out):
            outputs[name] = module_out
        return hook

    # Define layers
    layers = [3, 8, 13, 22] 
    # Define weights for layers
    layers_weights = [1,1,1,1,1] 

    # Register hook on each layer with index on array "layers"
    for layer in layers:
        handle = vgg[layer].register_forward_hook(save_output(layer))

    vgg(target)
    gramm_targets = [gramm(outputs[key]) for key in layers] 


    ######################################################################
    # Training

    # optimizer
    optimizer = optim.Adam(gen.parameters(), lr=learning_rate)
    if use_GN: 
        I = Normalize_gradients.apply
    loss_history = numpy.zeros(max_iter)

    sz = [int(img_size/1),int(img_size/2),int(img_size/4),int(img_size/8),int(img_size/16),int(img_size/32)]

    #run training
    for n_iter in range(max_iter):
        optimizer.zero_grad()

        # element by element to allow the use of large training sizes
        for i in range(batch_size):

            zk = [torch.rand(1,n_input_ch,szk,szk) for szk in sz]
            z_samples = [Variable(z.to(device)) for z in zk ]
            batch_sample = gen(z_samples)
            sample = batch_sample[0,:,:,:].unsqueeze(0)

            # Forward pass using sample. Get activations of selected layers for image sample (outputs).
            out = vgg(sample)
            sample_outputs = [outputs[key] for key in layers] 
            
            # Compute loss for each activation
            losses = []
            for activations in zip(sample_outputs, gramm_targets, layers_weights):
                losses.append(gram_loss(*activations).unsqueeze(0))
            
            total_loss = torch.cat(losses).sum()*(1/(batch_size))
            total_loss.backward()

            loss_history[n_iter] = loss_history[n_iter] + total_loss.item()

            del out, losses, total_loss, batch_sample, z_samples, zk

        print('Iteration: %d, loss: %f'%(n_iter, loss_history[n_iter]))
            
        optimizer.step()
        if n_iter%lr_adjust == (lr_adjust-1):
            optimizer.param_groups[0]['lr'] = lr_decay_coef * optimizer.param_groups[0]['lr']
            print('---> lr adjusted to '+str(optimizer.param_groups[0]['lr']))

    # save final model
    torch.save(gen.state_dict(),'./Trained_models/'+ input_name[:-4] +'/params.pytorch')

    return gen