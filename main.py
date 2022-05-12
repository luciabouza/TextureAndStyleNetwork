# TRAIN GENERATOR PYRAMID 2D PERIODIC
#
# Code for the texture synthesis and style transfer method in:
# Ulyanov et al. Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
# https://arxiv.org/abs/1603.03417 
# Generator architecture fixed to 6 scales!
#
# Author: Lucía Bouza
# Based on https://github.com/JorgeGtz/TextureNets_implementation

import torch
from network import Pyramid2D
import argparse
import time
from model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Texture Networks')

    # General  
    parser.add_argument('--input_texture',dest='input_name',help='texture to train the network', type=str,default=None)  
    parser.add_argument('--input_style',dest='input_style',help='texture to train the network', type=str,default=None) 
    parser.add_argument('--input_content',dest='input_content',help='texture to train the network', type=str,default=None) 
    parser.add_argument('--sample_size',dest='sample_size', help='sample_size', type=int,default=512) 
        
    # Train
    parser.add_argument('--train_texture',dest='train_texture',help='if True train the model',type=bool,default=False)
    parser.add_argument('--train_style',dest='train_style',help='if True train the model',type=bool,default=False)
    parser.add_argument('--train_data',dest='train_data',help='Path to training datas',type=str, default='/mnt/data/shared_datasets/imagenet/imagenet/imagenet/train')
    parser.add_argument('--max_iter',dest='max_iter', help='if max number of iteration for training (5000 texture, 20000 style)', type=int,default=5000)
    parser.add_argument('--batch_size',dest='batch_size', help='batch_size', type=int,default=8)
    parser.add_argument('--learning_rate',dest='learning_rate', help='learning_rate', type=float,default=0.001)
    parser.add_argument('--lr_adjust',dest='lr_adjust', help='lr_adjust', type=int,default=750)
    parser.add_argument('--lr_decay_coef',dest='lr_decay_coef', help='lr_decay_coef', type=float,default=0.66)
    parser.add_argument('--use_GN',dest='use_GN', help='use_GN', type=bool,default=False)

    # Inference  
    parser.add_argument('--pretrained',dest='pretrained',help='path to pretrained model folder', type=str,default=None) 
    parser.add_argument('--pretrained_style',dest='pretrained_style',help='path to pretrained model folder', type=str,default=None)
    parser.add_argument('--n_samples',dest='n_samples', help='n_samples', type=int,default=3)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_input_ch = 3

    if args.train_texture:

        if args.input_name is None:
            raise Exception('please provide an input texture: --Input_texture <texture on Textures folder >')
        
        # train network and measure time
        t0 = time.time()
        gen = trainTextureSynthesis(device = device, input_name = args.input_name, max_iter = args.max_iter, n_input_ch = n_input_ch, batch_size = args.batch_size, learning_rate = args.learning_rate, lr_adjust = args.lr_adjust, lr_decay_coef = args.lr_decay_coef, use_GN = args.use_GN)    
        print('Time spent training: ', time.time()-t0)

        # sample after Training
        for param in gen.parameters(): param.requires_grad = False
        sample(generator = gen, n_samples = args.n_samples, sample_size = args.sample_size, n_input_ch = 3, folder = './Trained_models/' + args.input_name[:-4], device = device)

    elif args.train_style:

        if args.input_style is None:
            raise Exception('please provide an input style: --input_style <style on Style folder >')
        if args.input_content is None:
            raise Exception('please provide an input content: --input_content <content on Content folder >')
        if args.train_data is None:
            raise Exception('please provide training data : --train_data <path to your training data >')
        
        # train network and measure time
        t0 = time.time()
        gen = trainStyleTransfer(device = device, train_data = args.train_data, input_style = args.input_style, input_content = args.input_content, max_iter = args.max_iter, n_input_ch = n_input_ch, batch_size = args.batch_size, learning_rate = args.learning_rate, lr_adjust = args.lr_adjust, lr_decay_coef = args.lr_decay_coef, use_GN = args.use_GN)
        print('Time spent training: ', time.time()-t0)

        # sample after Training
        for param in gen.parameters(): param.requires_grad = False
        sample_style(generator = gen, sample_size = args.sample_size, n_input_ch = 3, folder = './Trained_models/' + args.input_style[:-4] , device = device, input_content=args.input_content)

    elif args.pretrained:

        if args.pretrained is None:
            raise Exception('please provide a pretrained model: --pretrained=path_to_your_model')
        # Load model
        model_folder = args.pretrained
        generator = Pyramid2D(ch_step=8)
        generator.load_state_dict(torch.load('./' + model_folder + '/params.pytorch', map_location=torch.device(device)))
        generator.to(device)
        for param in generator.parameters():
            param.requires_grad = False

        # Draw sample
        t0 = time.time()
        sample(generator = generator, n_samples = args.n_samples, sample_size = args.sample_size, n_input_ch = 3, folder = './' + model_folder, device = device)
        print('Time spent sampling texture: ', time.time()-t0)

    elif args.pretrained_style:

        if args.pretrained_style is None:
            raise Exception('please provide a pretrained model: --pretrained_style=path_to_your_model')
        # Load model
        model_folder = args.pretrained_style
        generator = Pyramid2D(ch_step=8)
        generator.load_state_dict(torch.load('./' + model_folder + '/params.pytorch', map_location=torch.device(device)))
        generator.to(device)
        for param in generator.parameters():
            param.requires_grad = False

        # Draw sample
        t0 = time.time()
        sample_style(generator = generator, sample_size = args.sample_size, n_input_ch = 3, folder = './' + model_folder, device = device, input_content=args.input_content )
        print('Time spent sampling style: ', time.time()-t0)

    else:
        raise Exception('please provide a one of this flags: --train_texture, --train_style, --pretrained or --pretrained_style')

