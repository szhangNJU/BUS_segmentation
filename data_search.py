# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:25:29 2020

@author: zs
"""
import argparse
import torch
from torch.utils.data import DataLoader,Subset
#from unet import NestedUNet,UNet,DenseUNet,weights_init
from cla_seg import NestedUNet,UNet,DenseUNet,weights_init
from PIL import Image
from data import Data
import numpy as np
import random
from utils import cls_seg_loss,calc_loss,dice_loss,area_error,cfg,transformI,transformM
from sklearn.metrics import roc_auc_score
import csv

def train(mode='all'):

    model.apply(weights_init)

    bce_weight = cfg.bce_weight
    initial_lr = args.lr
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[30,35],gamma=0.1)
    val_loss = 1000
    cls_l =0
    seg_l = 0
    dice_ = 0
    TP_ = 0
    FP_ = 0
    FN_ = 0
    auc_ = 0
    se_ = 0
    sp_ = 0
    for epoch in range(args.epochs):
        model.train()
        for img,target in train_loader:
            img,target = img.to(device),target.to(device)
            output,cls = model(img)
            cls_loss,seg_loss,label = cls_seg_loss(output,cls,target)
            if mode=='cls':
                loss = cls_loss
            elif mode =='seg':
                loss = seg_loss
            else:
                loss = cls_loss + seg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        losses = 0
        cls_losses = 0
        seg_losses = 0
        dice_coeff = 0
        iou = 0
        TP = 0
        FP = 0
        FN = 0
        gt = torch.FloatTensor()
        out = torch.FloatTensor()
        gt,out = gt.to(device),out.to(device)
        with torch.no_grad():
            for img,target in valid_loader:
                img,target = img.to(device),target.to(device)
                output,cls = model(img)
                cls_loss,seg_loss,label = cls_seg_loss(output,cls,target)
                if mode=='cls':
                    loss = cls_loss
                elif mode=='seg':
                    loss = seg_loss
                else:
                    loss = cls_loss+seg_loss
                cls_losses += cls_loss.item()
                seg_losses += seg_loss.item()
                losses += loss.item()
                output = torch.sigmoid(output)
                cls  = torch.sigmoid(cls)
                out = torch.cat((out,cls),0)
                gt = torch.cat((gt,label),0)
                dice = 1 - dice_loss(output,target)
                dice_coeff += dice.item()
                tpr,fpr,fnr = area_error(output,target,reduction='None')
                TP += tpr.item()
                FP += fpr.item()
                FN += fnr.item()
        losses /= len(valid_loader)
        cls_losses /= len(valid_loader)
        seg_losses /= len(valid_loader)
        p_num = gt.sum().item()
        total = gt.size(0)
        if total == p_num:
            auc = 1
        else:
            auc = roc_auc_score(gt.cpu().numpy(),out.cpu().numpy())
        out[out<=0.5] = 0
        out[out>0.5] = 1
        tp = (out*gt).sum().item()
        tn = ((1-out)*(1-gt)).sum().item()
        total = gt.size(0)
        dice_coeff = dice_coeff / len(valid_loader) *total / p_num
        TP /= p_num
        FP /= p_num
        FN /= p_num
        print('epoch{} valid end, val_loss:{:.3f}, auc:{:.3f}, se:{:.3f}, sp:{:.3f}, dice_coeff:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.
          format(epoch,losses,auc,tp/p_num,tn/(total-p_num+1e-5),dice_coeff,TP,FP,FN))
        if losses < val_loss:
            val_loss = losses
            cls_l = cls_losses
            seg_l = seg_losses
            TP_ = TP
            FP_ = FP
            FN_ = FN
            dice_ = dice_coeff
            auc_ = auc
            se_ = tp/p_num
            sp_ = tn/(total-p_num+1e-5)
            torch.save(model.state_dict(),model_name)
        scheduler.step()
    return val_loss,cls_l,seg_l,auc_,se_,sp_,dice_,TP_,FP_,FN_

def test_():
    
    test_data = Data(cfg.test_dir,clip=args.clip,test=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=5)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    dice_coeff = 0
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for img,target in test_loader:
            img,target = img.to(device),target.to(device)
            output,cls = model(img)
            output = torch.sigmoid(output)
            dice = 1 - dice_loss(output,target)
            dice_coeff += dice.item()
            tpr,fpr,fnr = area_error(output,target)
            TP += tpr.item()
            FP += fpr.item()
            FN += fnr.item()
    dice_coeff /= len(test_loader)
    TP /= len(test_loader)
    FP /= len(test_loader)
    FN /= len(test_loader)
    return dice_coeff,TP,FP,FN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--bs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true')
parser.add_argument('--n', dest='no', help='use notumor data or not', action='store_true')
parser.add_argument('--c', dest='clip', help='clip image or not', action='store_true')
parser.add_argument('--net', dest='net', choices=['NestedUNet','UNet','DenseUNet'], default='DenseUNet', type=str)
parser.add_argument('--mode','-m', dest='mode', choices=['cls','seg','all'], default='all', type=str)
args = parser.parse_args()

#initial model,data_loader and train config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.model = args.net
model = eval(cfg.model)(in_ch=1,out_ch=1)
model_name = 'model_{}.t7'
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

    training_data = Data(cfg.image_dir,no=args.no,clip=args.clip,transformI = transformI,transformM = transformM)
    num_train = len(training_data)
    indices = list(range(num_train))
    batch_size = args.bs
    valid_size = args.valid_ratio
    valid_num = int(np.floor(valid_size * num_train))
    random_seed = 0
    random_seed = random.randint(0,100)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    logf = 'log_{}_{}_{}.csv'.format(args.net,args.mode,args.no)
    with open(logf,'w') as log_f:
        log_c = csv.writer(log_f)
        log_c.writerow(['val_loss','cls_loss','seg_loss','auc','se','sp','dice_coeff','TP','FP','FN'])
    for i in range(int(1/valid_size)):
        valid_idx, train_idx = indices[i*valid_num:(i+1)*valid_num], np.take(indices,np.concatenate((np.arange(0,i*valid_num),np.arange((i+1)*valid_num,num_train))))
        train_set = Subset(training_data,train_idx)
        valid_set = Subset(training_data,valid_idx)
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True)
        valid_loader = DataLoader(valid_set,batch_size=batch_size,num_workers=5,pin_memory=True)
        val_loss,cls_loss,seg_loss,auc,se,sp,dice,TP,FP,FN = train(mode = args.mode)
        print('val_loss:{:.3f}, dice_coeff:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.format(val_loss,dice,TP,FP,FN))
        with open(logf,'a') as log_f:
            log_c = csv.writer(log_f)
            log_c.writerow([val_loss,cls_loss,seg_loss,auc,se,sp,dice,TP,FP,FN])
