from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from measure import SegmentationMetric
from dataset import train_dataset, train_dataset_Inria
# from MACUNet import MACUNet
# from model.MANet import MANet
# from model.BANet import BANet
from model.UNet_3Plus_Atten import UNet_3Plus
from early_stopping import EarlyStopping
from tqdm import tqdm, trange
import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from datasetload import SemiDataset

niter = 100
class_num = 2
band = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = UNet_3Plus(band, class_num)
net.load_state_dict(torch.load('checkpoint/kitsap/kitsap_UNetAtt_NoST.pth'))
net.to(device)

test_path = '/root/workspace/DWW/datasets/Inria/kitsap_NoST/test/'
result_path = './result/kitsap_UNet_NoST'
if not os.path.exists(result_path):
    os.makedirs(result_path)

metric = SegmentationMetric(class_num)

def Iou(input, target, classes=1):
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou

test_datatset_ = SemiDataset(test_path)
test_loader = DataLoader(test_datatset_, batch_size=1, shuffle=True, num_workers=0)
start = time.time()


net.eval()
iou_all = []
i=1
test_data = tqdm(test_loader)
for i, (initial_image, semantic_image) in enumerate(test_data):
    # print(initial_image.shape)
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()

    # semantic_image_pred = model(initial_image)
    semantic_image_pred = net(initial_image).detach()
    #semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
    semantic_image_pred = semantic_image_pred.squeeze(0).argmax(dim=0)


    semantic_image = torch.squeeze(semantic_image.cpu(), 0)
    semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
    semantic_image_pred = np.array(semantic_image_pred).astype(uint8)
    semantic_image = np.array(semantic_image).astype(uint8)

    img = semantic_image
    pred = semantic_image_pred
    img[img==1]=255
    pred[pred==1]=255
    img = Image.fromarray(img)
    pred =Image.fromarray(pred)
    img.save(result_path+'/{}.png'.format(i))
    pred.save(result_path+'/{}_pred.png'.format(i))
    i=i+1




    iou = Iou(semantic_image_pred, semantic_image, 1)
    iou_all.append(iou)
    
# end = time.time()
# print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
# mIoU = metric.meanIntersectionOverUnion()
print('mIoU: ', np.mean(iou_all))