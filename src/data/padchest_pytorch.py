import os
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
import glob
import numpy as np
from src.config import PROCESSED_DATA_DIR

class PadchestDataset(Dataset):
    def __init__(self, split, dataset_name, group=None, as_tensor=False):
        self.labels_csv = pd.read_csv(f'{PROCESSED_DATA_DIR}/padchest/{split}_labels.csv',index_col=0)
        self.labels_csv = self.labels_csv[self.labels_csv["ImageID"].isin([p.split("/")[-1] for p in glob.glob(f'{PROCESSED_DATA_DIR}/padchest/images/*.png')])]

        self.group = group
        if self.group:
            group_label = group[0]
            group_value = group[1]
            self.labels_csv = self.labels_csv[self.labels_csv[group_label] == group_value]
        
        self.labels_csv = self.labels_csv.reset_index(drop=True)
        self.labels_csv["img_paths"] = [f"{PROCESSED_DATA_DIR}/padchest/images/{img_id}" for img_id in self.labels_csv["ImageID"]]
        self.labels["Report"] = self.labels["Report"].fillna("no report")

        self.as_tensor = as_tensor
        self.dataset_name = dataset_name
    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_row = self.labels_csv.iloc[idx]
        img_path = img_row["img_paths"]
        image = read_image(img_path,ImageReadMode.RGB)
        image = image / image.max()
        
        label = img_row["label"]
        if self.as_tensor:
            return torch.Tensor(image), img_row["Report"], torch.tensor(label), img_row["ImageID"], self.dataset_name
        else:
            return image.numpy(), img_row["Report"], label, img_row["ImageID"], self.dataset_name

    def get_image_id(self,idx):
        img_path = self.labels_csv.iloc[idx]["img_paths"]
        return img_path        

    def get_image_by_id(self,id):
        idx = np.where(np.array(self.labels_csv["img_paths"])==id)[0][0]
        return self.__getitem__(idx)


def get_padchest_datasets_to_evaluate():
    configs = {
        "All":None,
        "Female":["PatientSex_DICOM","F"],
        "Male":["PatientSex_DICOM","M"],
        "Philips":["Manufacturer_DICOM","PhilipsMedicalSystems"],
        "ImagingDynamics":["Manufacturer_DICOM","ImagingDynamicsCompanyLtd"],
    }
    lst_datasets = [PadchestDataset(split="train",dataset_name=config_name,group=configs[config_name],as_tensor=True) for config_name in configs]
    return lst_datasets

def get_padchest_test():
    return PadchestDataset(split="test",dataset_name="test",group=None,as_tensor=True)