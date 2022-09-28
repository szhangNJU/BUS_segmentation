# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:41:47 2020

@author: zs
"""
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

cfg = EasyDict()
cfg.input_size = (288,288) #size for image after processing
cfg.crop_size = (256,256)
cfg.image_dir = [] # train datasets path
cfg.test_dir = [] # test datasets path


cfg.bce_weight = 0.5 # default 0.5
cfg.pos_w = None
cfg.cls = 'focal' #choices = ['bce','focal'], class loss function

cfg.log_dir = './log/'
cfg.cp_dir = './checkpoint/'

def dice_loss(predict,target,reduction='mean',mode = 'log'):
    #mode = minus return 1-dicecoeffitient log return -log(dice_coefficient)
    smooth = 1
    num = predict.size(0)
    i_flat = predict.view(num,-1)
    t_flat = target.view(num,-1)

    intersection = (i_flat * t_flat).sum(1)
    #intersection[t_flat.sum(1)<=0] = 0.5*i_flat.sum(1)[t_flat.sum(1)<=0]*(1-i_flat.sum(1)[t_flat.sum(1)<=0].log())
    dice = (2. * intersection+smooth ) / (i_flat.sum(1) + t_flat.sum(1) + smooth)
    if mode == 'minus':
        dice = 1-dice
    else:
        dice = -dice.log() 
    if reduction=='mean':
        return dice.sum()/num
    else:
        return dice

def area_error(predict,target,threshold=0.5,reduction = 'mean'):
    num = predict.size(0)
    i_flat = predict.view(num,-1)
    t_flat = target.view(num,-1)
    ta = t_flat.sum(1)
    ta[ta<=0]=-1
    i_flat[i_flat<threshold]=0
    i_flat[i_flat>=threshold]=1
    tpa = (i_flat*t_flat).sum(1)
    fpa = (i_flat*(1-t_flat)).sum(1)
    fna = ((1-i_flat)*t_flat).sum(1)
    tpa[ta<0] = 0
    fpa[ta<0] = 0
    fna[ta<0] = 0
    num = (ta>0).sum()
    if num<=0: num=1
    iou = (tpa/(i_flat.sum(1)+t_flat.sum(1)+1e-5-tpa)).sum()/num
    TP = (tpa/ta).sum()/num
    FP = (fpa/ta).sum()/num
    FN = (fna/ta).sum()/num
    if reduction == 'mean':
        return iou,TP,FP,FN
    else:
        return iou*num,TP*num,FP*num,FN*num

def get_hausdorf(predict,target,reduction = 'mean'):
    #get hausdorf distance
    num = predict.size(0)
    h_d = target.new_zeros((num,1))
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    #hausdorff_sd.setRankProportion(0.9)
    for i in range(num):
        cnt_p, cnt_t = get_max_cnt(predict[i]),get_max_cnt(target[i])
        if cnt_p is not None and cnt_t is not None:
            h_d[i] = hausdorff_sd.computeDistance(cnt_p, cnt_t)
        else:
            h_d[i] = 0
    if reduction == 'mean':
        return h_d.mean()
    else:
        return h_d.sum()

def get_boundary_error(predict,target,reduction = 'mean'):
    #get mean absolute deviation and hausdorff distance
    num = predict.size(0)
    ad = target.new_zeros((num,1))
    hd = target.new_zeros((num,1))
    for i in range(num):
        cnt_p, cnt_t = get_max_cnt(predict[i]),get_max_cnt(target[i])
        if cnt_p is not None and cnt_t is not None:
            cnt_p = cnt_p.repeat(cnt_t.shape[0],1)
            cnt_t = cnt_t.swapaxes(0,1).repeat(cnt_p.shape[0],0)
            ad2 = np.sum(np.abs(cnt_p-cnt_t),2)
            ad_p = ad2.min(1).astype(np.float)
            ad_t = ad2.min(0).astype(np.float)
            ad[i] = 0.5*ad_p.mean()+0.5*ad_t.mean()
            hd[i] = max(ad_p.max(),ad_t.max())
        else:
            ad[i] = 0
            hd[i] = 0
    if reduction == 'mean':
        return ad.mean(),hd.mean()
    else:
        return ad.sum(),hd.sum()

def get_max_cnt(img):
    #get max contour 
    img[img>0.5] = 1
    img[img<=0.5] = 0
    img = img.byte().cpu().numpy().copy().squeeze()
    cnts, hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = [cv2.contourArea(cnt) for cnt in cnts]
    if area == []: return None
    max_idx = np.argmax(area)
    return cnts[max_idx]

def ig_mean(loss,label):
    t_num = label.sum()
    loss = (label.reshape(loss.shape)*loss).sum()
    if t_num>0:
        loss = loss/t_num
    else:
        loss = loss*0
    return loss


def cls_seg_loss(output,cls,target,bce_weight = 0.5, m = 0,mode = 'log'):
    pos_weight = cfg.pos_w#torch.Tensor([0.25]).type_as(cls)
    if pos_weight: pos_weight = pos_weight.type_as(cls)
    num = output.size(0)
    t_flat = target.view(num,-1)
    label = target.new_zeros((num,1))
    label[t_flat.sum(1)>0]=1
    if cfg.cls == 'bce':
        cls_loss = 0.5*F.binary_cross_entropy_with_logits(cls, label,pos_weight=pos_weight)
    else:
        cls_loss = focal_loss(cls,label)

    bce,dice = calc_loss(output,target,reduction='none',mode = mode)
    if m ==0:
        seg_loss = ig_mean(bce,label)*bce_weight+ig_mean(dice,label)*(1-bce_weight)
    elif m ==1:
        seg_loss = bce.mean()*bce_weight+ig_mean(dice,label)*(1-bce_weight)
    elif m ==2:
        seg_loss = bce.mean()*bce_weight+dice.mean()*(1-bce_weight)
    
    return cls_loss,seg_loss,label

def calc_loss(prediction, target, reduction='mean',mode = 'log'):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    #if cfg.cls == 'bce':
    bce = F.binary_cross_entropy_with_logits(prediction, target,reduction=reduction)
    #else:
    #    bce = focal_loss(prediction,target,reduction = reduction)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target, reduction,mode = mode)
    if reduction=='none':
        bce = bce.view(bce.size(0),-1)
        bce = bce.mean(1)

    return bce,dice

def focal_loss(y, t, gamma = 2, alpha = None, reduction='mean'):

    pt = torch.clamp(torch.sigmoid(y),1e-5,1-1e-5)
    loss1 = -t * (1-pt).detach()**gamma * torch.log(pt)
    loss0 = -(1-t) * pt.detach()**gamma * torch.log(1-pt)
    #loss1 = -t * (1-pt)**gamma * torch.log(pt)
    #loss0 = -(1-t) * pt**gamma * torch.log(1-pt)
    if alpha is not None:
        loss = alpha * loss1 + (1-alpha)*loss0
    else:
        loss = loss1 + loss0
    if reduction=='sum':
        loss = loss.sum()
    elif reduction=='mean':
        loss = loss.mean()
    return loss

def weights_init(m):

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight,a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

