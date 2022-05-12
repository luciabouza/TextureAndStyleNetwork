#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm 
from utilities import *


class Dataset(data.Dataset):

    def __init__(self,file,img_size=192):
        """
        Args:
            file (string): Repertory where the datas are.
            img_size (int): size of images for training.
            normalize (boolean): if True normalize images using imagenet statistics
        """
        super().__init__()
        self.img_files = list()
        for file in  tqdm(Path(file).glob('**/*.JPEG'),unit='files'): #tqdm plots a progress bar
            self.img_files.append(file)           
        self.size = len(self.img_files)
        self.img_size = img_size

    def __getitem__(self,index):
        index = index % self.size
        img = prep_img(image=self.img_files[index], size=self.img_size, both=True)
        return img
  
    def __len__(self):
        return self.size
