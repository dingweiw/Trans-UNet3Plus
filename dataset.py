import numpy as np
import torch
import torch.utils.data as data
import glob
import imageio
import os
from torchvision import transforms

def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg'])

def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = np.array(mask)
        mask[mask==255] = 1
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0, time_series=4, batch_size=1):
        super(train_dataset, self).__init__()
        self.src_list = np.array(sorted(glob.glob(data_path + 'image/' + '*.tif')))
        self.lab_list = np.array(sorted(glob.glob(data_path + 'label/' + '*.tif')))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def data_iter_index(self, index=1000):
        batch_index = np.random.choice(len(self.src_list), index)
        x_batch = self.src_list[batch_index]
        y_batch = self.lab_list[batch_index]
        data_series = []
        label_series = []
        try:
            for i in range(index):
                data_series.append(imageio.imread(x_batch[i]))
                label = imageio.imread(y_batch[i])
                label[label==255] = 1
                label_series.append(label)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

    def data_iter(self):
        data_series = []
        label_series = []
        try:
            for i in range(len(self.src_list)):
                data_series.append(imageio.imread(self.src_list[i]))
                label = imageio.imread(self.lab_list[i])
                label[label==255] = 1
                label_series.append(label)
                self.index += 1

        except OSError:
            return None, None

        data_series = torch.from_numpy(np.array(data_series)).type(torch.FloatTensor)
        data_series = data_series.permute(0, 3, 1, 2)
        label_series = torch.from_numpy(np.array(label_series)).type(torch.FloatTensor)
        torch_data = data.TensorDataset(data_series, label_series)
        data_iter = data.DataLoader(
            dataset=torch_data,  # torch TensorDataset format
            batch_size=self.batch_size,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

        return data_iter

class train_dataset_Inria(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0, time_series=4, batch_size=1):
        super(train_dataset_Inria, self).__init__()
        self.src_list = np.array(sorted(glob.glob(data_path + 'image/' + '*.tif')))
        self.lab_list = np.array(sorted(glob.glob(data_path + 'label/' + '*.tif')))
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip
        self.time_series = time_series
        self.index = 0
        self.batch_size = batch_size

    def __getitem__(self, item):
        image = self.src_list[item]
        label = self.lab_list[item]
        img = imageio.imread(image)
        label = imageio.imread(label)
        label[label==255]=1
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img.permute(2, 0, 1)
        label = torch.from_numpy(label).type(torch.FloatTensor)

       
        return img, label

    def __len__(self):
        return len(self.src_list)