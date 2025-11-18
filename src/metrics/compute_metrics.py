import enum
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR,INTERIM_DATA_DIR
from src.data.morphomnist_pytorch import MorphoMNISTDataset,get_thickening_datasets,get_thinning_datasets,get_perturb_dataset,get_test_dataset
from src.morphomnist import perturb
from src.modeling.utils import stratified_downsampling_dataset, bootstrap_resampling
import pandas as pd
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
from vendi_score import vendi
from skimage.feature import hog

from rouge_score import rouge_scorer

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
    if "inception_preds" not in dataset.orig.labels_csv.columns:
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            lst_preds.append(get_pred(batchv))
        preds = np.concatenate(lst_preds,axis=0)
        #Save the features for later use in bootstrap, if the dataset is a ConcatDataset save in the dataframes of the original datasets to be selected to be effective during bootstrap.
        if isinstance(dataset.orig,ConcatDataset):
            for i,d in enumerate(dataset.orig.datasets):
                d.labels_csv["inception_preds"]=pd.Series(preds[i*len(d.labels_csv):(i+1)*len(d.labels_csv)].tolist())
        else:
            dataset.orig.labels_csv["inception_preds"]=pd.Series(preds.tolist())

    else:
        preds = np.stack(dataset.orig.labels_csv["inception_preds"].values)
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

def get_inception_feature(x,resize=False):
    # Set up dtype
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # Load inception model and change the last layer to get the features before the last linear layer
    inception_model = inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    if resize:
        x = up(x)
    x = inception_model(x)
    return x.data.cpu().numpy()

def fid(dataset,reference_dataset,batch_size=32, resize=False,splits=1,load_ref_stats=False):
    
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

    # Compute inception features if not already in the dataset
    if "inception_features" not in dataset.orig.labels_csv.columns:
        lst_features_dataset = []
        for i, batch in enumerate(dataloader_data, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            lst_features_dataset.append(get_inception_feature(batchv,resize=resize))
        feat_data = np.concatenate(lst_features_dataset,axis=0)

        #Save the features for later use in bootstrap, if the dataset is a ConcatDataset save in the dataframes of the original datasets to be selected to be effective during bootstrap.
        if isinstance(dataset.orig,ConcatDataset):
            for i,d in enumerate(dataset.orig.datasets):
                d.labels_csv["inception_features"]=pd.Series(feat_data[i*len(d.labels_csv):(i+1)*len(d.labels_csv)].tolist())
        else:
            dataset.orig.labels_csv["inception_features"]=pd.Series(feat_data.tolist())
    else:
        feat_data = np.stack(dataset.orig.labels_csv["inception_features"].values)
    mu_data = np.mean(feat_data, axis=0)
    sigma_data = np.cov(feat_data, rowvar=False)

    if not load_ref_stats:
        lst_preds_reference = []
        for i, batch in enumerate(dataloader_ref, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            lst_preds_reference.append(get_inception_feature(batchv,resize=resize))
        feat_ref = np.concatenate(lst_preds_reference,axis=0)
        mu_ref = np.mean(feat_ref, axis=0)
        sigma_ref = np.cov(feat_ref, rowvar=False)
        with open(INTERIM_DATA_DIR/'fid_ref.npy', 'wb') as f:
            np.save(f,mu_ref)
            np.save(f,sigma_ref)
    else:
        with open(INTERIM_DATA_DIR/'fid_ref.npy', 'rb') as f:
            mu_ref = np.load(f)
            sigma_ref = np.load(f)

    #Compute the fid 
    mu_diff = np.sum((mu_data - mu_ref)**2.0)
    covmean = sqrtm(sigma_data.dot(sigma_ref))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return mu_diff + np.trace(sigma_data + sigma_ref - 2.0 * covmean)

def vs_pixels(dataset):
    """
    Generate features for a dataset using the pixel values. To be used in the Vendi Score.
    """
    imgs = [item[0] for item in dataset]
    if dataset.as_tensor:
        pixel_vectors = np.stack([np.array(img.detach().cpu().numpy()).flatten() for img in imgs], 0)
    else:
        pixel_vectors = np.stack([np.array(img).flatten() for img in imgs], 0)
    return pixel_vectors

def vs_hog(dataset):
    """
    Generate features for a dataset using the histogram of oriented grandients. To be used in the Vendi Score.
    """
    imgs = [item[0] for item in dataset]
    if dataset.as_tensor:
        hog_vectors = np.stack([np.array(hog(img.detach().cpu().numpy(),channel_axis=0)) for img in imgs], 0)
    else:
        hog_vectors = np.stack([np.array(hog(img,channel_axis=0)) for img in imgs], 0)
    return hog_vectors

def vs_inception_features(dataset):
    """
    Generate features for a dataset using the features of a pretrained InceptionV3 model. To be used in the Vendi Score.
    """
    dataset = IgnoreLabelDataset(dataset)
    # Set up dtype
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader_data = torch.utils.data.DataLoader(dataset, batch_size=32)

    # Get features
    if "inception_features" not in dataset.orig.labels_csv.columns:
        lst_features_dataset = []
        for i, batch in enumerate(dataloader_data, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            lst_features_dataset.append(get_inception_feature(batchv,resize=True))
        inception_features_vectors = np.concatenate(lst_features_dataset,axis=0)
    else:
        inception_features_vectors = np.stack(dataset.orig.labels_csv["inception_features"].values)

    return inception_features_vectors

def vendi_score(dataset,nb_resampling,feature_function):
    """
    Compute the vendi score of a dataset using the cosine similarity features generated by the feature_function.
    For computational reason, the function will downsample nb_resampling times the original dataset to keep 10% of the data and return the mean value.
    @param:
        - dataset: Pytorch dataset to be evaluated
        - nb_resampling: number of time the downsampling will be performed
        - feature_function: function to generate the features used in the computation of the cosine similarity
    """
    lst_vendi_scores = []
    for _ in range(nb_resampling):
        d_downsample = stratified_downsampling_dataset(dataset)
        feature_vectors = feature_function(d_downsample)
        n, d = feature_vectors.shape
        if n < d:
            pixel_vs = vendi.score_X(feature_vectors)
        else:
            pixel_vs = vendi.score_dual(feature_vectors)
        lst_vendi_scores.append(pixel_vs)
    
    return np.mean(lst_vendi_scores)


def rougeL(dataset):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    texts = [item[1] for item in dataset]
    for i in range(len(texts)):
        t1 = texts[i]
        for j in range(i+1,len(texts)):
            t2 = texts[j]
            rouge_scores.append(scorer.score(t1,t2).fmeasure) 
    return np.mean(rouge_scores)

def get_confidence_interval(values,alpha=5.0):
    alpha = 5.0
    lower_p = alpha / 2.0
    lower = np.percentile(values, lower_p)
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = np.percentile(values, upper_p)
    return lower,upper

def evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path,nb_bootstrap=2):
    #Create csv file with header
    with open(res_file_path,"w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join([ds.dataset_name for ds in lst_train_datasets])}")

    #Lists of metrics values and confidence interval each value will be MetricOnFullDataset_LowerCI_UpperCI
    lst_is = []
    lst_fid = []
    lst_vs_pix = []
    lst_vs_hog = []
    lst_vs_inception = []

    for dataset in tqdm(lst_train_datasets):
        #Compute each metric on the dataset
        logger.info("compute on full")
        is_dataset = inception_score(dataset,32,True,1)[0]
        fid_dataset = fid(dataset,ref_dataset,32,True,1)
        vs_pix_dataset = vendi_score(dataset,1,vs_pixels)
        vs_hog_dataset = vendi_score(dataset,1,vs_hog)
        vs_inception_dataset = vendi_score(dataset,1,vs_inception_features)

        #List of metrics value on bootstrap version of the dataset
        lst_bootstrap_is = []
        lst_bootstrap_fid = []
        lst_bootstrap_vs_pix = []
        lst_bootstrap_vs_hog = []
        lst_bootstrap_vs_inception = []

        for i in range(nb_bootstrap):
            logger.info(f"compute on bootstrap {i}")
            d_bootstrap = bootstrap_resampling(dataset)
            is_bootstrap = inception_score(d_bootstrap,32,True,1)[0]
            fid_bootstrap = fid(d_bootstrap,ref_dataset,32,True,1,True)
            vs_pix_bootstrap = vendi_score(d_bootstrap,5,vs_pixels)
            vs_hog_bootstrap = vendi_score(d_bootstrap,5,vs_hog)
            vs_inception_bootstrap = vendi_score(d_bootstrap,5,vs_inception_features)

            lst_bootstrap_is.append(is_bootstrap)
            lst_bootstrap_fid.append(fid_bootstrap)
            lst_bootstrap_vs_pix.append(vs_pix_bootstrap)
            lst_bootstrap_vs_hog.append(vs_hog_bootstrap)
            lst_bootstrap_vs_inception.append(vs_inception_bootstrap)

        #Compute the 95% confidence interval from the boostrap values
        l_is,u_is = get_confidence_interval(lst_bootstrap_is)
        l_fid,u_fid = get_confidence_interval(lst_bootstrap_fid)
        l_vs_pix,u_vs_pix = get_confidence_interval(lst_bootstrap_vs_pix)
        l_vs_hog,u_vs_hog = get_confidence_interval(lst_bootstrap_vs_hog)
        l_vs_inception,u_vs_inception = get_confidence_interval(lst_bootstrap_vs_inception)

        #Add the metric and CI
        lst_is.append(f"{is_dataset}_{l_is}_{u_is}")
        lst_fid.append(f"{fid_dataset}_{l_fid}_{u_fid}")
        lst_vs_pix.append(f"{vs_pix_dataset}_{l_vs_pix}_{u_vs_pix}")
        lst_vs_hog.append(f"{vs_hog_dataset}_{l_vs_hog}_{u_vs_hog}")
        lst_vs_inception.append(f"{vs_inception_dataset}_{l_vs_inception}_{u_vs_inception}")
    
    #Write the result in the csv file
    with open(res_file_path,"a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")
        metrics_csvfile.write(f"\nfid,{','.join([str(fid_d) for fid_d in lst_fid])}")
        metrics_csvfile.write(f"\nvs_pixel,{','.join([str(vs_d) for vs_d in lst_vs_pix])}")    
        metrics_csvfile.write(f"\nvs_hog,{','.join([str(vs_d) for vs_d in lst_vs_hog])}")    
        metrics_csvfile.write(f"\nvs_inception,{','.join([str(vs_d) for vs_d in lst_vs_inception])}")    

@app.command()
def main():    
    ref_dataset = get_test_dataset()

    # # ---- Thinning evolution ----
    # logger.info("Computing diversity metrics for multiple thinning parameter...")
    # lst_train_datasets = get_thinning_datasets()
    # res_file_path = INTERIM_DATA_DIR / "thinning_diversity_metrics_local.csv"
    # evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)

    # logger.success("Done.")
    # # -----------------------------------------
    
    # # ---- Thickening evolution ----
    # logger.info("Computing diversity metrics for multiple thickening parameter...")
    # lst_train_datasets = get_thickening_datasets()
    # res_file_path = INTERIM_DATA_DIR / "thickening_diversity_metrics_local.csv"
    # evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)
    # logger.success("Done.")
    # # -----------------------------------------


    # ---- Multiple scenarios ----
    logger.info("Computing diversity metrics for multiple scenarios...")
    lst_train_datasets = get_perturb_dataset()
    res_file_path = INTERIM_DATA_DIR / "diversity_metrics.csv"
    evaluate_datasets(lst_train_datasets,ref_dataset,res_file_path)
    logger.success("Done.")
    # -----------------------------------------

if __name__ == "__main__":
    app()