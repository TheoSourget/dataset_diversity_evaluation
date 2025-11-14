from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import ConcatDataset
torch.manual_seed(1907)
np.random.seed(1907)

def stratified_downsampling_dataset(dataset,frac=0.1):
    """
    Downsample the provided dataset in a stratified way, keeping 10% of the data for each class
    """
    #If dataset is a ConcatDataset, need to apply the resampling to all the dataset within it
    if isinstance(dataset,ConcatDataset):
        lst_datasets=[]
        for d in dataset.datasets:
            d_downsample = deepcopy(d)
            idxs = d_downsample.labels_csv.groupby('label', group_keys=False)["label"].apply(lambda x: x.sample(frac=frac)).index
            d_downsample.labels_csv = d_downsample.labels_csv.iloc[idxs].reset_index(drop=True)
            d_downsample.imgs = d_downsample.imgs[idxs]
            lst_datasets.append(d_downsample)
        d_downsample = ConcatDataset(lst_datasets)
        d_downsample.as_tensor = dataset.datasets[0].as_tensor
    else:
        d_downsample = deepcopy(dataset)
        idxs = d_downsample.labels_csv.groupby('label', group_keys=False)["label"].apply(lambda x: x.sample(frac=frac)).index
        d_downsample.labels_csv = d_downsample.labels_csv.iloc[idxs].reset_index(drop=True)
        d_downsample.imgs = d_downsample.imgs[idxs]
    return d_downsample

def bootstrap_resampling(dataset):
    """
    Resample a dataset using the bootstrap method (selection of the same number of sample as in the original dataset but with replacement)
    """
    if isinstance(dataset,ConcatDataset):
        lst_datasets=[]
        idxs = np.random.randint(0, len(dataset.datasets[0].labels_csv), len(dataset.datasets[0].labels_csv)*len(dataset.datasets))
        for i,d in enumerate(dataset.datasets):
            d_bootstrap = deepcopy(d)
            idxs_d = idxs[i*len(dataset.datasets[0].labels_csv):(i+1)*len(dataset.datasets[0].labels_csv)]
            d_bootstrap.labels_csv = d_bootstrap.labels_csv.iloc[idxs_d].reset_index(drop=True)
            d_bootstrap.imgs = d_bootstrap.imgs[idxs_d]
            lst_datasets.append(d_bootstrap)
        d_bootstrap = ConcatDataset(lst_datasets)
        d_bootstrap.as_tensor = dataset.datasets[0].as_tensor
    else:
        d_bootstrap = deepcopy(dataset)
        idxs = np.random.randint(0, len(d_bootstrap.labels_csv), len(d_bootstrap.labels_csv))
        d_bootstrap.labels_csv = d_bootstrap.labels_csv.iloc[idxs].reset_index(drop=True)
        d_bootstrap.imgs = d_bootstrap.imgs[idxs]
    return d_bootstrap