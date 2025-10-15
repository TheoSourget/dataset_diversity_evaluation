import pandas as pd
from src.morphomnist.skeleton import num_neighbours
import torch
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import Dataset,ConcatDataset
from src.morphomnist import morpho
import numpy as np
from copy import deepcopy

from src.config import PROCESSED_DATA_DIR
from src.morphomnist import perturb

class MorphoMNISTDataset(Dataset):
    def __init__(self, split, dataset_name, morpho_transforms, as_tensor=False):
        """
        Create an instance of the MorphoMNIST dataset
        @param:
            - split: str, "train" or "test", define if using "train" or "test" split
            - dataset_name: str, Name to associate with this instance of the dataset (e.g. plain or thin)
            - morpho_transforms: list, morpho transformation to apply to the images, use None to use the base images
            - as_tensor: bool, Define if the __getitem__ returns the images as a tensor or as a numpy array
        """
        self.morpho_transforms = morpho_transforms
        self.labels_csv = pd.read_csv(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/labels.csv")
        self.labels_csv["dataset_name"] = dataset_name
        self.labels_csv["img_path"] = self.labels_csv["img_id"].apply(lambda img_id: PROCESSED_DATA_DIR / f"morphomnist/{split}/{img_id}")
        self.imgs = np.array([self.__load_img__(path) for path in self.labels_csv["img_path"]])
        self.dataset_name = dataset_name
        self.as_tensor = as_tensor
        

    def __load_img__(self,path):
        image = decode_image(path,ImageReadMode.GRAY).detach().cpu().numpy()[0,:,:]
        
        if self.morpho_transforms:
            #Apply the morpho transformation
            morphology = morpho.ImageMorphology(deepcopy(image), scale=4)
            for pertubation in self.morpho_transforms:
                image = pertubation(morphology)
            image= morphology.downscale(image)

        #normalize the image and convert to (3,h,w)
        image = image / 255
        image = np.stack([image]*3, axis=0)

        return image

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        #Get sample info in the csv
        img_row = self.labels_csv.iloc[idx]
        
        #Get a copy of the image so that potential later transformations are not applied to the original image
        image = deepcopy(self.imgs[idx])

        if self.as_tensor:
            #Return image and label as Tensor
            return torch.Tensor(image), torch.tensor(img_row["label"])
        else:
            #Return image as numpy array and label as str
            return image,img_row["label"]


def get_thinning_datasets():
    """
    Instantiate a list of MorphoMNISTDataset with various Thinning parameter (0-1, increase of 0.1)
    """
    config_datasets = {
        "thin_0": None,
        "thin_10": [perturb.Thinning(amount=0.1)],
        "thin_20": [perturb.Thinning(amount=0.2)],
        "thin_30": [perturb.Thinning(amount=0.3)],
        "thin_40": [perturb.Thinning(amount=0.4)],
        "thin_50": [perturb.Thinning(amount=0.5)],
        "thin_60": [perturb.Thinning(amount=0.6)],
        "thin_70": [perturb.Thinning(amount=0.7)],
        "thin_80": [perturb.Thinning(amount=0.8)],
        "thin_90": [perturb.Thinning(amount=0.9)],
        "thin_100": [perturb.Thinning(amount=1)],
    }
    
    #List of datasets with a single configuration
    lst_train_datasets = [MorphoMNISTDataset("train",dataset_name,morpho_transforms=config_datasets[dataset_name],as_tensor=True) for dataset_name in config_datasets]
    return lst_train_datasets

def get_thickening_datasets():
    """
    Instantiate a list of MorphoMNISTDataset with various Thickening parameter (0-1, increase of 0.1)
    """
    config_datasets = {
        "thick_0": None,
        "thick_10": [perturb.Thickening(amount=0.1)],
        "thick_20": [perturb.Thickening(amount=0.2)],
        "thick_30": [perturb.Thickening(amount=0.3)],
        "thick_40": [perturb.Thickening(amount=0.4)],
        "thick_50": [perturb.Thickening(amount=0.5)],
        "thick_60": [perturb.Thickening(amount=0.6)],
        "thick_70": [perturb.Thickening(amount=0.7)],
        "thick_80": [perturb.Thickening(amount=0.8)],
        "thick_90": [perturb.Thickening(amount=0.9)],
        "thick_100": [perturb.Thickening(amount=1)],
    }
    
    #List of datasets with a single configuration
    lst_train_datasets = [MorphoMNISTDataset("train",dataset_name,morpho_transforms=config_datasets[dataset_name],as_tensor=True) for dataset_name in config_datasets]
    return lst_train_datasets

def get_perturb_dataset():
    config_datasets = {
        "plain": None,
        "thin": [perturb.Thinning(amount=0.5)],
        "thick": [perturb.Thickening(amount=0.5)],
        "swelling": [perturb.Swelling(strength=3, radius=7,random_seed=1907)],
        "fracture": [perturb.Fracture(num_frac=3,random_seed=1907)],
    }
    
    #List of datasets with a single configuration
    lst_train_datasets = [MorphoMNISTDataset("train",dataset_name,morpho_transforms=config_datasets[dataset_name],as_tensor=True) for dataset_name in config_datasets]
    combined_datasets = []
    #Add combination of multiple datasets
    for d1 in lst_train_datasets:
        for d2 in lst_train_datasets:
            if (d1.dataset_name == 'plain') and (d2.dataset_name != 'plain'):
                concat_dataset = ConcatDataset([d1,d2])
                concat_dataset.dataset_name=f"plain_{d2.dataset_name}"
                concat_dataset.as_tensor = d1.as_tensor
                combined_datasets.append(concat_dataset)
    lst_train_datasets = lst_train_datasets+ combined_datasets
    return lst_train_datasets

def get_test_dataset():
    return MorphoMNISTDataset("test","test",morpho_transforms=None,as_tensor=True)