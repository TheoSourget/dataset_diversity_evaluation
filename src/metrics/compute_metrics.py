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
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from scipy.linalg import sqrtm

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
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

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
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get features
    lst_features_dataset = []
    for i, batch in enumerate(dataloader_data, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        lst_features_dataset.append(get_pred(batchv))

    lst_preds_reference = []
    for i, batch in enumerate(dataloader_ref, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        lst_preds_reference.append(get_pred(batchv))

    feat_data = np.concatenate(lst_features_dataset,axis=0)
    feat_ref = np.concatenate(lst_preds_reference,axis=0)

    mu_data = np.mean(feat_data, axis=0)
    sigma_data = np.cov(feat_data, rowvar=False)
    
    mu_ref = np.mean(feat_ref, axis=0)
    sigma_ref = np.cov(feat_ref, rowvar=False)

    print(mu_data,mu_ref)
    mu_diff = np.sum((mu_data - mu_ref)**2.0)
    covmean = sqrtm(sigma_data.dot(sigma_ref))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return mu_diff + np.trace(sigma_data + sigma_ref - 2.0 * covmean)

def vendi_score(dataset,distance_metric):
    pass



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
    with open(INTERIM_DATA_DIR / "thinning_diversity_metrics.csv","w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join([ds.dataset_name for ds in lst_train_datasets])}")
    lst_is = []
    lst_fid = []
    for dataset in tqdm(lst_train_datasets):
        is_dataset = inception_score(dataset,32,True,1)[0]
        fid_dataset = fid(dataset,ref_dataset,32,True,1)
        lst_is.append(is_dataset)
        lst_fid.append(fid_dataset)
    with open(INTERIM_DATA_DIR / "thinning_diversity_metrics.csv","a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")
        metrics_csvfile.write(f"\nfid,{','.join([str(fid_d) for fid_d in lst_fid])}")

    logger.success("Done.")
    # -----------------------------------------
    
    # ---- Thickening evolution ----
    logger.info("Computing diversity metrics for multiple thickening parameter...")
    lst_train_datasets = get_thickening_datasets()
    with open(INTERIM_DATA_DIR / "thickening_diversity_metrics.csv","w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join([ds.dataset_name for ds in lst_train_datasets])}")
    lst_is = []
    lst_fid = []
    for dataset in tqdm(lst_train_datasets):
        is_dataset = inception_score(dataset,32,True,1)[0]
        fid_dataset = fid(dataset,ref_dataset,32,True,1)
        lst_is.append(is_dataset)
        lst_fid.append(fid_dataset)
    with open(INTERIM_DATA_DIR / "thickening_diversity_metrics.csv","a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")
        metrics_csvfile.write(f"\nfid,{','.join([str(fid_d) for fid_d in lst_fid])}")
    logger.success("Done.")
    # -----------------------------------------


    # ---- Multiple scenarios ----
    logger.info("Computing diversity metrics for multiple scenarios...")
    lst_train_datasets = get_perturb_dataset()
    with open(INTERIM_DATA_DIR / "diversity_metrics.csv","w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join([ds.dataset_name for ds in lst_train_datasets])}")
    lst_is = []
    lst_fid = []
    for dataset in tqdm(lst_train_datasets):
        is_dataset = inception_score(dataset,32,True,1)[0]
        fid_dataset = fid(dataset,ref_dataset,32,True,1)
        lst_is.append(is_dataset)
        lst_fid.append(fid_dataset)
    with open(INTERIM_DATA_DIR / "diversity_metrics.csv","a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")
        metrics_csvfile.write(f"\nfid,{','.join([str(fid_d) for fid_d in lst_fid])}")
    logger.success("Done.")
    # -----------------------------------------

if __name__ == "__main__":
    app()