import torch
import torch.utils.data as data
import os
import cv2

class Dataset(data.Dataset):
    def __init__(self, dataset_dir):
        super(Dataset, self).__init__()

        dir_list = os.listdir(dataset_dir)
        self.path_list = [os.path.join(dataset_dir, filename) for filename in dir_list] 
        
    def __getitem__(self, index):
        x = cv2.imread(self.path_list[index])
        x = x.transpose(2, 0, 1)
        self.x_data = torch.from_numpy(x).float()
        return self.x_data

    def __len__(self):
        return len(self.path_list)
