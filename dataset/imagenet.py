import h5py
import torch
import torch.utils.data as data
import numpy as np
import time


class Imagenet(data.Dataset):
    def __init__(self, imagenet_path):
        super(Imagenet, self).__init__()

        self.h5file = h5py.File(imagenet_path, 'r')
        self.num = self.h5file['images'].shape[0]
        self.crop_size = 256
        np.random.seed(int(time.time()))
        
    def __getitem__(self, index):
        x = self.h5file['images'][index]
        shape = self.h5file['shapes'][index]
        x = np.reshape(x, shape)
        x = self.random_crop_numpy(x)
        self.x_data = torch.from_numpy(x).float()
        return self.x_data

    def __len__(self):
        return self.num

    def random_crop_numpy(self, img):
        y = np.random.randint(img.shape[1] - self.crop_size + 1)
        x = np.random.randint(img.shape[2] - self.crop_size + 1)
        return img[:, y : y+self.crop_size, x : x+self.crop_size]

