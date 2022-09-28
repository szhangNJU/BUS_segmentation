# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:25:29 2020

@author: zs
"""
import argparse
import torch
from torch.utils.data import DataLoader,Subset
from unet import NestedUNet,UNet,DenseUNet,weights_init
from PIL import Image
from data import Data
import numpy as np
import random
from utils import calc_loss,dice_loss,area_error,get_hausdorf,get_boundary_error,cfg,transformI,transformM
import csv

def train(size):

    model.apply(weights_init)
    training_data = Data(cfg.image_dir,no=args.no,clip=args.clip,transformI = transformI,transformM = transformM)
    batch_size = args.bs
    valid_size = args.valid_ratio
    num_train = len(training_data)
    data_size = num_train
    indices = list(range(num_train))
    split = int(np.floor(valid_size * data_size))
    random_seed = 0
    random_seed = random.randint(0,100)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:data_size], indices[:split]
    train_set = Subset(training_data,train_idx)
    valid_set = Subset(training_data,valid_idx)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,num_workers=5,pin_memory=True)

    bce_weight = cfg.bce_weight
    initial_lr = args.lr
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[30,35],gamma=0.1)
    val_loss = 1000
    for epoch in range(args.epochs):
        model.train()
        losses = 0
        for img,target in train_loader:
            img,target = img.to(device),target.to(device)
            output = model(img)
            loss = calc_loss(output,target,bce_weight)
            losses += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        model.eval()
        losses = 0
        dice_coeff = 0
        haus = 0
        TP = 0
        FP = 0
        FN = 0
        iou = 0
        with torch.no_grad():
            for img,target in valid_loader:
                img,target = img.to(device),target.to(device)
                output = model(img)
                loss = calc_loss(output,target,bce_weight)
                losses += loss.item()
                output = torch.sigmoid(output)
                dice = 1 - dice_loss(output,target)
                dice_coeff += dice.item()
                iou_,tpr,fpr,fnr = area_error(output,target)
                h = get_hausdorf(output,target)
                haus += h.item()
                iou += iou_.item()
                TP += tpr.item()
                FP += fpr.item()
                FN += fnr.item()
        losses /= len(valid_loader)
        dice_coeff /= len(valid_loader)
        haus /= len(valid_loader)
        iou /= len(valid_loader)
        TP /= len(valid_loader)
        FP /= len(valid_loader)
        FN /= len(valid_loader)
        print('epoch{} valid end, val_loss:{:.3f}, dice_coeff:{:.3f}, iou{:.3f}, haus:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.
          format(epoch,losses,dice_coeff,iou,haus,TP,FP,FN))
        if losses < val_loss:
            val_loss = losses
            torch.save(model.state_dict(),model_name)
        scheduler.step()
    return val_loss,dice_coeff,TP,FP,FN

def test():
    
    test_data = Data(cfg.test_dir,clip=args.clip,test=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=5)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    dice_coeff = 0
    iou = 0
    TP = 0
    FP = 0
    FN = 0
    haus = 0
    mad = 0
    hau = 0
    with torch.no_grad():
        for img,target in test_loader:
            img,target = img.to(device),target.to(device)
            output = model(img)
            output = torch.sigmoid(output)
            dice = 1 - dice_loss(output,target)
            dice_coeff += dice.item()
            iou_,tpr,fpr,fnr = area_error(output,target)
            ad,ha = get_boundary_error(output,target)
            h = get_hausdorf(output,target)
            iou += iou_.item()
            haus += h.item()
            mad += ad.item()
            hau += ha.item()
            TP += tpr.item()
            FP += fpr.item()
            FN += fnr.item()
            
    dice_coeff /= len(test_loader)
    iou /= len(test_loader)
    haus /= len(test_loader)
    mad /= len(test_loader)
    hau /= len(test_loader)
    TP /= len(test_loader)
    FP /= len(test_loader)
    FN /= len(test_loader)
    return dice_coeff,iou,haus,mad,hau,TP,FP,FN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--bs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true')
parser.add_argument('--n', dest='no', help='use notumor data or not', action='store_true')
parser.add_argument('--c', dest='clip', help='clip image or not', action='store_true')
args = parser.parse_args()

#initial model,data_loader and train config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = eval(cfg.model)(in_ch=1,out_ch=1)
model_name = 'grid_search{}.t7'
if args.no and args.clip:
    model_name=model_name.format('_no_clip')
elif args.no:
    model_name=model_name.format('_no')
elif args.clip:
    model_name=model_name.format('_clip')
else:
    model_name=model_name.format('')
if args.resume: model.load_state_dict(torch.load(model_name))
model = model.to(device)
    
if __name__ == '__main__':

    f = 'log{}.csv'
    if args.no and args.clip:
        f=f.format('_no_clip')
    elif args.no:
        f=f.format('_no')
    elif args.clip:
        f=f.format('_clip')
    else:
        f=f.format('')
    with open(f,'w') as log_f:
        log_c = csv.writer(log_f)
        log_c.writerow(['size','val_loss','dice_coeff','hausdorff','TP','FP','FN'])
    '''for i in range(20):
      for size in [100,400,800,1163]:#1000,1487]:
        val_loss,dice,TP,FP,FN = train(size)
        dice_coeff,haus,TP,FP,FN = test()
        print('size:{} test, val_loss:{:.3f}, dice_coeff:{:.3f}, haus:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.format(size,val_loss,dice_coeff,haus,TP,FP,FN))
        with open(f,'a') as log_f:
            log_c = csv.writer(log_f)
            log_c.writerow([size,val_loss,dice_coeff,acc,tpa,fpa,TP,FP,FN])'''
    train(0)
    dice_coeff,iou,haus,mad,hau,TP,FP,FN = test()
    print('test end, dice_coeff:{:.3f}, iou:{:.3f}, haus:{:.3f}, mad:{:.3f}, hau:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.format(dice_coeff,iou,haus,mad,hau,TP,FP,FN))
    
