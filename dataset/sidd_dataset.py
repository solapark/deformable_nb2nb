import torch
import torch.utils.data as data
import os
import cv2

class SIDD_dataset(data.Dataset):
    def __init__(self, noisy_dataset_dir,clean_dataset_dir):
        super(SIDD_dataset, self).__init__()

        noisy_dir_list = sorted(os.listdir(noisy_dataset_dir))
        clean_dir_list = sorted(os.listdir(clean_dataset_dir))
        self.noisy_path_list = [os.path.join(noisy_dataset_dir, filename) for filename in noisy_dir_list] 
        self.clean_path_list = [os.path.join(clean_dataset_dir, filename) for filename in clean_dir_list] 
        
    def __getitem__(self, index):
        x = cv2.imread(self.noisy_path_list[index])
        y = cv2.imread(self.clean_path_list[index])
        x = x.transpose(2, 0, 1)
        y = y.transpose(2, 0, 1)
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(y).float()
        return {'noisy':self.x_data, 'gt':self.y_data}

    def __len__(self):
        return len(self.noisy_path_list)
