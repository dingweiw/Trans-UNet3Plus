from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from numpy import *
from measure import SegmentationMetric
from dataset import train_dataset
from datasetload import SemiDataset
# from MACUNet import MACUNet
# from model.MANet import MANet
# from model.BANet import BANet
# from model.MAResUNet import MAResUNet
from model.UNet_3Plus_Atten import UNet_3Plus
from early_stopping import EarlyStopping
from tqdm import tqdm, trange
import wandb
from torch.utils.data import DataLoader
batch_size = 4
niter = 100
class_num = 2
learning_rate = 0.0001 * 3
beta1 = 0.5
cuda = True
num_workers = 1
size_h = 256
size_w = 256
flip = 0
band = 3
net = UNet_3Plus(band, class_num)
train_path = '/root/workspace/DWW/datasets/Inria/tyrol-w_NoST/train/'
val_path = '/root/workspace/DWW/datasets/Inria/tyrol-w_NoST/val/'
test_path = '/root/workspace/DWW/datasets/Inria/tyrol-w_NoST/test/'
out_file = './checkpoint/tyrol-w/'
num_GPU = 1
index = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'tyrol-w_UNetAtt_NoST.pth'
try:
    import os
    os.makedirs(out_file)
except OSError:
    pass

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
cudnn.benchmark = True

train_datatset_ = SemiDataset(train_path)
train_loader = DataLoader(train_datatset_, batch_size=batch_size, shuffle=True, num_workers=0)
val_datatset_ = SemiDataset(val_path)
val_loader = DataLoader(val_datatset_, batch_size=1, shuffle=True, num_workers=0)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

try:
    os.makedirs(out_file)
    os.makedirs(out_file + '/')
except OSError:
    pass
if cuda:
    net.to(device=device)
if num_GPU > 1:
    net = nn.DataParallel(net)


###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
metric = SegmentationMetric(class_num)
early_stopping = EarlyStopping(patience=10, verbose=True)

if __name__ == '__main__':
    # wan_db = wandb.init(project='UNet-Attention', resume='allow')
    # wan_db.config.update(dict(epoches=niter, batch_size=batch_size, learning_rate=learning_rate))
    start = time.time()
    net.train()
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1)
    for epoch in range(1, niter + 1):
        tbar = tqdm(train_loader)
        for i, (initial_image, semantic_image) in enumerate(tbar):
            initial_image = initial_image.cuda()
            semantic_image = semantic_image.cuda()

            semantic_image_pred = net(initial_image)

            loss = criterion(semantic_image_pred, semantic_image.long())
            loss_train_cpu = loss.to('cpu')
            # wan_db.log({'loss_train':loss_train_cpu})


            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_adjust.step()
        with torch.no_grad():
            net.eval()
            val_iter = tqdm(val_loader)

            for i, (initial_image, semantic_image) in enumerate(val_iter):
                # print(initial_image.shape)
                initial_image = initial_image.cuda()
                semantic_image = semantic_image.cuda()

                semantic_image_pred = net(initial_image).detach()
                loss_val = criterion(semantic_image_pred, semantic_image.long())

                loss_val_cpu = loss_val.to('cpu')
                # wan_db.log({
                #     'loss_val':loss_val_cpu
                # })

                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)

                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

                metric.addBatch(semantic_image_pred, semantic_image)

        mIoU = metric.meanIntersectionOverUnion()
        print('mIoU: ', mIoU)
        # wan_db.log({
        #     'mIou':mIoU
        # })
        metric.reset()
        net.train()

        early_stopping(1 - mIoU, net, '%s/' % out_file + model_name)

        if early_stopping.early_stop:
            break

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

    test_datatset_ = SemiDataset(test_path)
    test_loader = DataLoader(test_datatset_, batch_size=1, shuffle=True, num_workers=0)
    start = time.time()
    test_iter = tqdm(test_loader)
    if os.path.exists('%s/' % out_file + model_name):
        net.load_state_dict(torch.load('%s/' % out_file + model_name))

    net.eval()
    for i, (initial_image, semantic_image) in enumerate(test_iter):
        # print(initial_image.shape)
        initial_image = initial_image.cuda()
        semantic_image = semantic_image.cuda()

        # semantic_image_pred = model(initial_image)
        semantic_image_pred = net(initial_image).detach()
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)

        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)

        metric.addBatch(semantic_image_pred, semantic_image)
        image = semantic_image_pred

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    mIoU = metric.meanIntersectionOverUnion()
    print('mIoU: ', mIoU)