import enum
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR,INTERIM_DATA_DIR
from src.data.morphomnist_pytorch import MorphoMNISTDataset
from src.morphomnist import perturb

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from scipy.stats import entropy


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

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
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
    inception_model.eval();
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

def vendi_score(dataset,distance_metric):
    pass

def fid(dataset,reference_dataset):
    pass


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # -----------------------------------------
):
    metrics_csv_path = INTERIM_DATA_DIR / "diversity_metrics.csv"

    # ---- Creating the datasets ----
    logger.info("Instanciating the datasets...")

    config_train_datasets = {
        "plain": None,
        "thin": [perturb.Thinning(amount=0.5)],
        "thick": [perturb.Thickening(amount=0.5)]
    }
    
    lst_train_datasets = [MorphoMNISTDataset("train",dataset_name,morpho_transforms=config_train_datasets[dataset_name],as_tensor=True) for dataset_name in config_train_datasets]
    # -----------------------------------------
    
    with open(metrics_csv_path,"w") as metrics_csvfile:
        metrics_csvfile.write(f"metric_name,{','.join(config_train_datasets.keys())}")

    # ---- Inception score ----
    logger.info("Computing inception scrores...")
    lst_is = []
    for dataset in tqdm(lst_train_datasets):
        is_dataset = inception_score(dataset,32,True,1)[0]
        lst_is.append(is_dataset)
    with open(metrics_csv_path,"a+") as metrics_csvfile:
        metrics_csvfile.write(f"\ninception_score,{','.join([str(is_d) for is_d in lst_is])}")

    logger.success("Inception scores computed.")
    # -----------------------------------------

    # ---- FID ----


if __name__ == "__main__":
    app()