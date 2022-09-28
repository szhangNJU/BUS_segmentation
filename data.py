# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:31:45 2020

@author: zs
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import glob
from utils import cfg
from PIL import Image
import numpy as np

train_transform = A.Compose(
 [
  A.Resize(*cfg.crop_size),
  #A.Equalize(p=1),
  A.HorizontalFlip(p=0.5),
  A.PadIfNeeded(min_height = cfg.input_size[0],min_width = cfg.input_size[1],border_mode = 4, value = 0,mask_value=0),
  A.RandomSizedCrop(min_max_height=(cfg.crop_size[0]-20, cfg.crop_size[0]), height=cfg.crop_size[0], width=cfg.crop_size[1]),
  A.Rotate(limit=10),
  A.ShiftScaleRotate(p=0.4, border_mode=0, shift_limit=0.04, scale_limit=0.03, rotate_limit = 10),
  #A.ISONoise(p=0.5),
  #A.Perspective(p=0.5),
  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
  #A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,p=0.5),
  A.Normalize(mean=[0.5],std=[0.5]),
  ToTensorV2(),
 ]
)
class Data(Dataset):
    def __init__(self,img_path='',no=False,clip=False,test=False,transform=None):
        c='tumor'
        if clip: c+='_clip'
        images=[]
        for path in img_path:
            images += glob.glob(path+'/'+c+'/*.png')
            if no:
                 images += glob.glob(path+'/no'+c+'/*.png')
        self.imgs_dir = [img for img in images if img.endswith('1.png')]
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(*cfg.crop_size),
                #A.Equalize(p=1),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
                ])

    def __len__(self):
        return len(self.imgs_dir)
    def __getitem__(self, index):
        img_path = self.imgs_dir[index]
        img = Image.open(img_path).convert('L')
        target = Image.open(img_path[:-5]+'t.png').convert('L')
        img = np.array(img)
        target = np.array(target)
        if np.max(target) == 255:
            target = target/255
        target = np.where(target>0.5,1.0,0)
        target = target.astype(np.float32)
        transformed = self.transform(image = img,mask = target)
        img = transformed['image']
        target = transformed['mask']
        target = target.unsqueeze(0)
        return img,target
