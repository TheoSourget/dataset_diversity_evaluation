import pandas as pd
import torch
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import Dataset
from src.morphomnist import morpho
import numpy as np
from copy import deepcopy

from src.config import PROCESSED_DATA_DIR

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
        self.labels_csv = pd.read_csv(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/labels.csv")
        self.labels_csv["dataset_name"] = dataset_name
        self.labels_csv["img_path"] = self.labels_csv["img_id"].apply(lambda img_id: PROCESSED_DATA_DIR / f"morphomnist/{split}/{img_id}")
        self.dataset_name = dataset_name
        self.as_tensor = as_tensor
        self.morpho_transforms = morpho_transforms
        

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        #Get sample info in the csv
        img_row = self.labels_csv.iloc[idx]
        
        #Load the image
        image = decode_image(img_row["img_path"],ImageReadMode.GRAY)
        image = image.detach().cpu().numpy()[0,:,:]
        if self.morpho_transforms:
            #Apply the morpho transformation
            morphology = morpho.ImageMorphology(deepcopy(image), scale=4)
            for pertubation in self.morpho_transforms:
                image = pertubation(morphology)
            image= morphology.downscale(image)

        #normalize the image and convert to (3,h,w)
        image = image / 255
        image = np.stack([image]*3, axis=0)

        if self.as_tensor:
            #Return image and label as Tensor 
            return torch.Tensor(image), torch.Tensor(img_row["label"])
        else:
            #Return image as numpy array and label as str
            return image,img_row["label"]
    