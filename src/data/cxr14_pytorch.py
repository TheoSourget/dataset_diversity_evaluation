import os
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import glob
import numpy as np
from src.config import PROCESSED_DATA_DIR

class CXR14Dataset(Dataset):
    def __init__(self):
        self.labels_csv = pd.read_csv(f'{PROCESSED_DATA_DIR}/CXR14/processed_labels.csv',index_col=0)        
        self.labels_csv = self.labels_csv.reset_index(drop=True)
        self.labels_csv["img_paths"] = [f"{PROCESSED_DATA_DIR}/CXR14/imgs/{img_id}" for img_id in self.labels_csv["Image Index"]]

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_row = self.labels_csv.iloc[idx]
        img_path = img_row["img_paths"]
        image = read_image(img_path,ImageReadMode.RGB)
        image = image / image.max()
        label = img_row["Pneumothorax"]
        return torch.Tensor(image), torch.tensor(label), img_row["Image Index"]

    def get_image_id(self,idx):
        img_path = self.labels_csv.iloc[idx]["img_paths"]
        return img_path        

    def get_image_by_id(self,id):
        idx = np.where(np.array(self.labels_csv["Image Index"])==id)[0][0]
        return self.__getitem__(idx)