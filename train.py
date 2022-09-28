# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:25:29 2020

@author: zs
"""
import argparse
import torch
from torch.utils.data import DataLoader,Subset,SubsetRandomSampler
from unet import NestedUNet,UNet,DenseUNet
from cenet import CENet
from cpfnet import CPFNet
from fpnn import FPNN
from PIL import Image
from data import Data, train_transform
import numpy as np
import random
from utils import calc_loss,dice_loss,cfg,weights_init,area_error,get_boundary_error#,get_hausdorf
import csv
from sklearn.metrics import roc_auc_score,roc_curve,auc
from torchsummary import summary

def train():

    initial_lr = args.lr
    opt = torch.optim.Adam(model.parameters(), lr=initial_lr)
    #opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
     #                     lr=0.005, momentum=0.9, weight_decay=0.0001)
    #opt = torch.optim.ASGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[int(args.epochs*0.5),int(args.epochs*0.75)],gamma=0.1)
    val_loss = -1
    for epoch in range(args.epochs):
        model.train()
        losses = 0
        for img,target in train_loader:
            img,target = img.to(device),target.to(device)
            output = model(img)
            bce,dice = calc_loss(output,target)
            loss = cfg.bce_weight*bce+dice*(1-cfg.bce_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += loss.item()
            del(loss)
        losses /= len(train_loader)
        print('epoch {} train end, loss:{:.3f}'.format(epoch,losses))

        model.eval()
        losses = 0
        dice_coeff = 0
        n_d = 0
        dd = 0
        iou = 0
        TP = 0
        FP = 0
        FN = 0
        gt = torch.FloatTensor()
        gt = gt.to(device)

        with torch.no_grad():
            for img,target in valid_loader:
                img,target = img.to(device),target.to(device)
                output = model(img)
                bce,dice = calc_loss(output,target)
                loss = bce*cfg.bce_weight+dice*(1-cfg.bce_weight)
                losses += loss.item()
                output = torch.sigmoid(output)
                num = output.size(0)
                t_flat = target.view(num,-1)
                label = target.new_zeros((num,1))
                label[t_flat.sum(1)>0]=1
                gt = torch.cat((gt,label),0)
                dice = 1 - dice_loss(output,target,mode='minus',reduction='None')
                di = 1 - dice_loss(output,target,mode='minus')
                dice_coeff += (label.reshape(dice.shape)*dice).sum().item()
                n_d += ((1-label).reshape(dice.shape)*dice).sum().item()
                dd += di.item()
                iou_,tpr,fpr,fnr = area_error(output,target,reduction='None')
                iou += iou_.item()
                TP += tpr.item()
                FP += fpr.item()
                FN += fnr.item()
        p_num = gt.sum().item()
        total = gt.size(0)
        if total != p_num:
            n_d = n_d / (total - p_num)
        dice_coeff = dice_coeff / p_num
        iou /= p_num
        TP /= p_num
        FP /= p_num
        FN /= p_num
        dd /= len(valid_loader)
        losses /= len(valid_loader)
        print('epoch{} valid end, val_loss:{:.3f}, dice:{:.3f}, dice_coeff:{:.3f}, n_d:{:.3f}, iou:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}'.
          format(epoch,losses,dd,dice_coeff,n_d,iou,TP,FP,FN))
        if dice_coeff > val_loss:
            val_loss = dice_coeff
            print('max valid DSC increased to {}, model saved'.format(val_loss))
            torch.save(model.state_dict(),model_name)
        scheduler.step()
        if not args.record: test()

def test(data = 0):
    if data==0:
        test_dir = cfg.test_dir[0:1]
    elif data == 1:
        test_dir = cfg.test_dir[1:]
    elif data == 2:
        test_dir = cfg.test_dir
    test_data = Data(test_dir,no = True, clip=not args.clip,test=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=5)
    print(len(test_loader))
    if args.record:
        model.load_state_dict(torch.load(model_name,map_location=torch.device(device)))
    model.eval()
    losses = 0
    dd = 0
    dice_coeff = 0
    n_d = 0
    iou = 0
    TP = 0
    FP = 0
    FN = 0
    t = 0
    num = 0
    #haus = 0
    mad = 0
    hau = 0
    global max_dice
    gt = torch.FloatTensor()
    gt = gt.to(device)
    with torch.no_grad():
        for img,target in test_loader:
            img,target = img.to(device),target.to(device)
            output = model(img)
            num = output.size(0)
            t_flat = target.view(num,-1)
            label = target.new_zeros((num,1))
            label[t_flat.sum(1)>0]=1
            bce,dice = calc_loss(output,target)
            loss = bce*cfg.bce_weight+dice*(1-cfg.bce_weight)
            losses += loss.item()
            output = torch.sigmoid(output)
            gt = torch.cat((gt,label),0)
            dice = 1 - dice_loss(output,target,mode='minus',reduction=None)
            di = 1 - dice_loss(output,target,mode='minus')
            dice_coeff += (label.reshape(dice.shape)*dice).sum().item()
            n_d += ((1-label).reshape(dice.shape)*dice).sum().item()
            iou_,tpr,fpr,fnr = area_error(output,target,reduction=None)
            ad,ha = get_boundary_error(output,target,reduction=None)
            #h = get_hausdorf(output,target)
            dd += di.item()
            iou += iou_.item()
            #haus += h.item()
            mad += ad.item()
            hau += ha.item()
            TP += tpr.item()
            FP += fpr.item()
            FN += fnr.item()

    p_num = gt.sum().item()
    total = gt.size(0)
    if total != p_num:
        n_d /= (total - p_num)
    dice_coeff = dice_coeff / p_num
    iou /= p_num
    TP /= p_num
    FP /= p_num
    FN /= p_num
    mad /= p_num
    hau /= p_num
    dd /= len(test_loader)
    losses /= len(test_loader)
    if not args.record and dice_coeff > max_dice:
        max_dice = dice_coeff
        print('max dice increased to {}, model saved'.format(max_dice))
        torch.save(model.state_dict(),cfg.cp_dir+cfg.model+'.t7') 
    
    return dd,dice_coeff,n_d,iou,mad,hau,TP,FP,FN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--bs', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--shuffle', '-s', action='store_true', help = 'shuffle to split the train and valid or not')
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--bw', type=float, default=0.5)
parser.add_argument('--r', dest='resume', help='resume checkpoint or not', action='store_true')
parser.add_argument('--re', dest='record', help='record test result or not', action='store_true')
parser.add_argument('--n', dest='no', help='use notumor data or not', action='store_true')
parser.add_argument('--c', dest='clip', help='clip image or not', action='store_true')
parser.add_argument('--mode','-m', dest='mode', choices=['cls','seg','all'], default='all', type=str)
parser.add_argument('--d', dest='d', choices=[0,1,2], default=2, type=int)
parser.add_argument('--sm', dest='sm', choices=[0,1,2], default=0, type=int)
parser.add_argument('--bm', dest='bm', choices=['minus','log'], default='log', type=str)
parser.add_argument('--loss', dest='cls', choices=['bce','focal'], default='focal', type=str)
parser.add_argument('--net', dest='net', choices=['UNet','DenseUNet','NestedUNet','UNet3Plus','CENet','CPFNet','FPNN'], default='DenseUNet', type=str)
args = parser.parse_args()

#initial model,data_loader and train config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
cfg.model = args.net
if args.d == 1:
  cfg.pos_w = torch.Tensor([0.25])
model = eval(cfg.model)(in_ch=1,out_ch=1)
model_name = cfg.cp_dir+cfg.model+'{}{}.t7'.format(args.d,args.no)
#model_name = cfg.cp_dir+cfg.model+'{}.t7'.format(args.cls)
#print(cfg.model)
cfg.cls = args.cls
cfg.bce_weight = args.bw
model = model.to(device)
#summary(model,(1,256,256))
#for m in model.children():
 # print(m)
if args.resume: model.load_state_dict(torch.load(model_name))

if args.d==0:
    train_dir = cfg.image_dir[0:1]
elif args.d == 1:
    train_dir = cfg.image_dir[1:2]
elif args.d == 2:
    train_dir = cfg.image_dir
training_data = Data(train_dir,no = not args.no,clip = not args.clip,transform = train_transform)
num_train = len(training_data)
indices = list(range(num_train))
batch_size = args.bs
valid_size = args.valid_ratio
valid_num = int(np.floor(valid_size * num_train))
random_seed = 0
random_seed = random.randint(0,100)
np.random.seed(random_seed)
np.random.shuffle(indices)
for i in range(int(1/valid_size)):
    valid_idx, train_idx = indices[i*valid_num:(i+1)*valid_num], np.take(indices,np.concatenate((np.arange(0,i*valid_num),np.arange((i+1)*valid_num,num_train))))
    train_set = Subset(training_data,train_idx)
    valid_set = Subset(training_data,valid_idx)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,num_workers=5,pin_memory=True)

    max_dice = -1
    train()
    import time
    start = time.time()
    for d in range(2):
        start = time.time()
        dd,dice_coeff,n_d,iou,mad,hau,TP,FP,FN = test(d)
        dur = time.time()-start
        print('test end, time:{}s, dd:{:.3f}, dice_coeff:{:.3f}, neg_dice:{:.3f}, iou:{:.3f}, mad:{:.3f}, hau:{:.3f}, TP:{:.3f}, FP:{:.3f}, FN:{:.3f}, AER:{:.3f}'.format(dur,dd,dice_coeff,n_d,iou,mad,hau,TP,FP,FN,FP+FN))
        if args.record:
            with open(cfg.log_dir+'log_{}{}{}{}.csv'.format(args.net,args.no,args.d,cfg.bce_weight),'a') as log_f:
            #with open(cfg.log_dir+'log_{}.csv'.format(cfg.bce_weight),'a') as log_f:
                log_c = csv.writer(log_f)
                log_c.writerow([dur,d,dd,dice_coeff,n_d,iou,mad,hau,TP,FP,FN,FP+FN])

