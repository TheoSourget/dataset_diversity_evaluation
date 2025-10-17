import enum
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR,INTERIM_DATA_DIR
from src.data.morphomnist_pytorch import MorphoMNISTDataset,get_thickening_datasets,get_thinning_datasets,get_perturb_dataset,get_test_dataset
from src.morphomnist import perturb

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models
from torchvision.models.inception import inception_v3
from torch.utils.data import ConcatDataset
from scipy.stats import entropy
from scipy.linalg import sqrtm
from vendi_score import vendi, image_utils

from copy import deepcopy

torch.manual_seed(1907)
np.random.seed(1907)

app = typer.Typer()

class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)
        
def inception_score(dataset, batch_size=32, resize=False, splits=1):
    """
    From https://github.com/sbarratt/inception-score-pytorch
    Computes the inception score of the generated images imgs

    dataset -- Torch dataset of (3xHxW) images normalized in the range [0, 1]
    batch_size -- batch size for feeding into Inception v3
    resize -- True if images need to be resized to 299x299 to fit the inception model
    splits -- number of splits
    """
    dataset = IgnoreLabelDataset(dataset)
    N = len(dataset)
    assert batch_size > 0

    # Set up dtype
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x,dim=-1).data.cpu().numpy()

    # Get predictions
    lst_preds = []

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        lst_preds.append(get_pred(batchv))
    preds = np.concatenate(lst_preds,axis=0)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def fid(dataset,reference_dataset,batch_size=32, resize=False, splits=1):
    
    dataset = IgnoreLabelDataset(dataset)
    reference_dataset = IgnoreLabelDataset(reference_dataset)

    assert batch_size > 0

    # Set up dtype
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    dataloader_ref = torch.utils.data.DataLoader(reference_dataset, batch_size=batch_size)

    # Load inception model and change the last layer to get the features before the last linear layer
    inception_model = inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_feature(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return x.data.cpu().numpy()

    # Get features
    lst_features_dataset = []
    for i, batch in enumerate(dataloader_data, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        lst_features_dataset.append(get_feature(batchv))

    lst_preds_reference = []
    for i, batch in enumerate(dataloader_ref, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        lst_preds_reference.append(get_feature(batchv))

    feat_data = np.concatenate(lst_features_dataset,axis=0)
    feat_ref = np.concatenate(lst_preds_reference,axis=0)

    mu_data = np.mean(feat_data, axis=0)
    sigma_data = np.cov(feat_data, rowvar=False)
    
    mu_ref = np.mean(feat_ref, axis=0)
    sigma_ref = np.cov(feat_ref, rowvar=False)

    mu_diff = np.sum((mu_data - mu_ref)**2.0)
    covmean = sqrtm(sigma_data.dot(sigma_ref))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return mu_diff + np.trace(sigma_data + sigma_ref - 2.0 * covmean)


def stratified_downsampling_dataset(dataset):
    #If dataset is a ConcatDataset, need to apply the resampling to all the dataset within it
    if isinstance(dataset,ConcatDataset):
        lst_datasets=[]
        for d in dataset.datasets:
            d_downsample = deepcopy(d)
            idxs = d_downsample.labels_csv.groupby('label', group_keys=False)["label"].apply(lambda x: x.sample(frac=0.1)).index
            d_downsample.labels_csv = d_downsample.labels_csv.iloc[idxs].reset_index(drop=True)
            d_downsample.imgs = d_downsample.imgs[idxs]
            lst_datasets.append(d_downsample)
        d_downsample = ConcatDataset(lst_datasets)
    else:
        d_downsample = deepcopy(dataset)
        idxs = d_downsample.labels_csv.groupby('label', group_keys=False)["label"].apply(lambda x: x.sample(frac=0.1)).index
        d_downsample.labels_csv = d_downsample.labels_csv.iloc[idxs].reset_index(drop=True)
        d_downsample.imgs = d_downsample.imgs[idxs]
    return d_downsample

def bootstrap_resampling(dataset):
    if isinstance(dataset,ConcatDataset):
        lst_datasets=[]
        idxs = np.random.randint(0, len(dataset.datasets[0].labels_csv), len(dataset.datasets[0].labels_csv)*len(dataset.datasets))
        for i,d in enumerate(dataset.datasets):
            d_bootstrap = deepcopy(d)
            idxs_d = idxs[i*len(dataset.datasets):(i+1)*len(dataset.datasets)]
            d_bootstrap.labels_csv = d_bootstrap.labels_csv.iloc[idxs_d].reset_index(drop=True)
            d_bootstrap.imgs = d_bootstrap.imgs[idxs_d]
            lst_datasets.append(d_bootstrap)
        d_bootstrap = ConcatDataset(lst_datasets)
    else:
        d_bootstrap = deepcopy(dataset)
        idxs = np.random.randint(0, len(d_bootstrap.labels_csv), len(d_bootstrap.labels_csv))
        d_bootstrap.labels_csv = d_bootstrap.labels_csv.iloc[idxs].reset_index(drop=True)
        d_bootstrap.imgs = d_bootstrap.imgs[idxs]
    return d_bootstrap

def vendi_score(dataset,nb_resampling):
    lst_vendi_scores = []
    for _ in range(nb_resampling):
        d_downsample = stratified_downsampling_dataset(dataset)
        imgs = [img for img,label in d_downsample]
        if dataset.as_tensor:
            pixel_vectors = np.stack([np.array(img.detach().cpu().numpy()).flatten() for img in imgs], 0)
        else:
            pixel_vectors = np.stack([np.array(img).flatten() for img in imgs], 0)
        n, d = pixel_vectors.shape
        if n < d:
            pixel_vs = vendi.score_X(pixel_vectors)
        else:
            pixel_vs = vendi.score_dual(pixel_vectors)
        lst_vendi_scores.append(pixel_vs)
    
    return np.mean(lst_vendi_scores)

def get_confidence_interval(values,alpha=5.0):
    alpha = 5.0
    lower_p = alpha / 2.0
    lower = np.percentile(values, lower_p)
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = np.percentile(values, upper_p)
    return lower,upper

def evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path,nb_bootstrap=1000):
    with open(res_file_path,"w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join([ds.dataset_name for ds in lst_train_datasets])}")
    lst_is = []
    lst_fid = []
    lst_vs = []
    for dataset in tqdm(lst_train_datasets):
        is_dataset = inception_score(dataset,32,True,1)[0]
        fid_dataset = fid(dataset,ref_dataset,32,True,1)
        vs_dataset = vendi_score(dataset,5)

        lst_bootstrap_is = []
        lst_bootstrap_fid = []
        lst_bootstrap_vs = []
        for i in range(nb_bootstrap):
            d_bootstrap = bootstrap_resampling(dataset)
            is_bootstrap = inception_score(d_bootstrap,32,True,1)[0]
            fid_bootstrap = fid(d_bootstrap,ref_dataset,32,True,1)
            vs_bootstrap = vendi_score(d_bootstrap,5)
            lst_bootstrap_is.append(is_bootstrap)
            lst_bootstrap_fid.append(fid_bootstrap)
            lst_bootstrap_vs.append(vs_bootstrap)

        l_is,u_is = get_confidence_interval(lst_bootstrap_is)
        l_fid,u_fid = get_confidence_interval(lst_bootstrap_fid)
        l_vs,u_vs = get_confidence_interval(lst_bootstrap_vs)

        lst_is.append(f"{is_dataset}_{l_is}_{u_is}")
        lst_fid.append(f"{fid_dataset}_{l_fid}_{u_fid}")
        lst_vs.append(f"{vs_dataset}_{l_vs}_{u_vs}")

    with open(res_file_path,"a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")
        metrics_csvfile.write(f"\nfid,{','.join([str(fid_d) for fid_d in lst_fid])}")
        metrics_csvfile.write(f"\nvs,{','.join([str(vs_d) for vs_d in lst_vs])}")    

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # -----------------------------------------
):    
    ref_dataset = get_test_dataset()

    # ---- Thinning evolution ----
    logger.info("Computing diversity metrics for multiple thinning parameter...")
    lst_train_datasets = get_thinning_datasets()
    res_file_path = INTERIM_DATA_DIR / "thinning_diversity_metrics_local.csv"
    evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)

    logger.success("Done.")
    # -----------------------------------------
    
    # ---- Thickening evolution ----
    logger.info("Computing diversity metrics for multiple thickening parameter...")
    lst_train_datasets = get_thickening_datasets()
    res_file_path = INTERIM_DATA_DIR / "thickening_diversity_metrics_local.csv"
    evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)
    logger.success("Done.")
    # -----------------------------------------


    # ---- Multiple scenarios ----
    logger.info("Computing diversity metrics for multiple scenarios...")
    lst_train_datasets = get_perturb_dataset()
    res_file_path = INTERIM_DATA_DIR / "diversity_metrics_local.csv"
    evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)
    logger.success("Done.")
    # -----------------------------------------

if __name__ == "__main__":
    app()