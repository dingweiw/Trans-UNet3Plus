import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import glob
import torch
def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.347, 0.379, 0.326], [0.170, 0.145, 0.134]),
    ])(img)
    mask = np.array(mask)
    mask[mask==255] = 1
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask

class SemiDataset(Dataset):
    def __init__(self, data_path=''):
        self.src_list = np.array(sorted(glob.glob(data_path + 'image/' + '*.tif')))
        self.lab_list = np.array(sorted(glob.glob(data_path + 'label/' + '*.tif')))


    def __getitem__(self, item):
        id = self.src_list[item]
        label_id = self.lab_list[item]
        img = Image.open(id)
        label = Image.open(label_id)
        img, mask = normalize(img, label)
        return img, mask



    def __len__(self):
        return len(self.src_list)
