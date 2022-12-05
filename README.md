# Introduction

### This repository is for the PAPER: *"Fully automatic tumor segmentation of breast ultrasound images with deep learning"*    published in the    ***Journal of Applied Clinical Medical Physics***.



## Citation
Zhang S, Liao M, Wang J, et al. Fully automatic tumor segmentation of breast ultrasound images with deep learning. J Appl Clin Med Phys. 2022;(00):e13863. https://doi.org/10.1002/acm2.13863


## Information

cla_seg.py: the proposed models.

ctrain.py: training code for the proposed models.

unet.py, cenet.py, cpfnet.py, fpnn.py: segmentation models for comparison.

train.py: training code for the above models.

utils.py: loss function and evaluation function, etc. 



## Requirements

sklearn\
pytorch\
torchvision\
albumentations\
opencv\
easydict
