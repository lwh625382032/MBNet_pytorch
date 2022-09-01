import argparse
import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils import data_augmentation


parser = argparse.ArgumentParser(description="DnCNN_dataset")
parser.add_argument("--color", type=bool, default=False, help="color or gray")
opt = parser.parse_args()


def normalize(data):
    return data / 255.


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]

    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]

    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])



def prepare_data(data_path, patch_size, stride, aug_times=1):
    # gray train
    print('\n process training data \n')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'BRDNet gray train 4744', '*.png'))  ### 训练彩色图像时候，要将 gray train 改成 color gray train
    files.sort()
    h5f = h5py.File('BRDNet gray train5050.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        if opt.color:    ### 彩色图
           img_bgr= cv2.imread(files[i])   ### 得到的图片为BGR格式
           img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    ## 转化成RGB格式

        else:          #### 灰度图
           img = cv2.imread(files[i])

        h, w, c = img.shape

        for k in range(len(scales)):

            Img = cv2.resize(img, (int(h * scales[k]), int(w * scales[k])), interpolation=cv2.INTER_CUBIC)
            if opt.color:
                Img = np.transpose(Img, (2, 0, 1))                #   彩色图预处理

            else:
                Img = np.expand_dims(Img[:,:,0].copy(), 0)        #   灰度图预处理


            Img = np.float32(normalize(Img))

            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d " % (files[i], scales[k], patches.shape[3] * aug_times))

            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num) + "_aug_%d" % (m + 1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, "Set12", '*.png'))  ### 训练彩色图像时，要将 Set12  改成 McMaster
    files.sort()
    h5f = h5py.File('BRDNet gray val5050.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])

        if opt.color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))                      # 彩色图预处理
        else:
            img = np.expand_dims(img[:, :, 0].copy(), 0)            # 灰度图预处理

        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples:   %d\n' % train_num)
    print('training set, # patch_size:  %d\n' % patch_size)
    print('val set, # samples:   %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('BRDNet gray train5050.h5', 'r')
        else:
            h5f = h5py.File('BRDNet gray val5050.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('BRDNet gray train5050.h5', 'r')
        else:
            h5f = h5py.File('BRDNet gray val5050.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)


