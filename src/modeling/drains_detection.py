"""
Generate the Drain label for the whole NIH-CXR14 dataset using annotation from the NEATX dataset
"""
import typer
from typing import Annotated

import cv2
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from torchvision.models import densenet121
from torchvision.io import read_image, ImageReadMode


from sklearn.metrics import roc_auc_score
from torch.nn.functional import sigmoid
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from loguru import logger

from src.data.padchest_pytorch import get_padchest_datasets_to_evaluate,get_padchest_test

import radt
import mlflow

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

torch.manual_seed(1907)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Define the Pytorch dataset
"""
class TubesDataset(Dataset):
    def __init__(self, labels,data_path):
        self.labels = labels
        self.data_path = data_path

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.labels.iloc[idx]
        image = read_image(f"{self.data_path}/imgs/{sample['Image Index']}",ImageReadMode.RGB)
        image = torch.Tensor(image)
        
        label = sample["Drain"]
        label = torch.Tensor([label])
        return image, label
    
def training_epoch(model,criterion,optimizer,train_dataloader):
    model.to(DEVICE)
    model.train()
    train_loss = 0.0
    lst_labels = []
    lst_probas = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        output_sigmoid = sigmoid(outputs)
        lst_labels.extend(labels.cpu().detach().numpy())
        lst_probas.extend(output_sigmoid.cpu().detach().numpy())
    lst_labels = np.array(lst_labels)
    lst_probas = np.array(lst_probas)
    auc_score=roc_auc_score(lst_labels,lst_probas)
    return train_loss,auc_score

def test_model(model,valid_dataloader):
    model.to(DEVICE)
    model.eval()
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(np.array(labels)).float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)
        auc_score=roc_auc_score(lst_labels,lst_probas)
    return auc_score

def get_padchest_preds(model,padchest_dataloader):
    model.to(DEVICE)
    model.eval()
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(padchest_dataloader, 0):
            inputs = data[0]
            inputs = inputs.float().to(DEVICE)
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs)
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
        lst_probas = np.array(lst_probas)
    return lst_probas


@app.command()
def main(
    train: Annotated[bool, typer.Option(help=("--train to train model on CXR14, --no-train to apply it on PadChest data"))],
    nb_epochs: int=200,
    batch_size: int=32,
    lr: float=0.0001,
    data_path: str=f"{PROCESSED_DATA_DIR}/CXR14",
    weights: str=f"{MODELS_DIR}/tube_detection_model.pt",
):  
    """
    Train or apply a Densenet-121 model to detect chest drains in chest x-rays
    
    :param train: Train the model with CXR14 data if True (--train), Apply to PadChest data if False (--no-train)
    :type train: bool\n
    :param nb_epochs: Number of training epochs
    :type nb_epochs: int\n
    :param batch_size: Batch size for the data loader
    :type batch_size: int\n
    :param lr: Learning rate for the optimizer
    :type lr: float\n
    :param data_path: Path to the folder containing the images and label csv
    :type data_path: str\n
    :param weights: Path to the weights to use, if "DEFAULT" use the imagenet-1k pretrained weights, otherwise load the weights from the given path
    :type weights: str\n
    """
    model = densenet121(weights='DEFAULT')
    kernel_count = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
     torch.nn.Flatten(),
     torch.nn.Linear(kernel_count, 1)
    )
    if weights != "DEFAULT":
        model.load_state_dict(torch.load(weights,map_location=torch.device('cpu')))
    model.to(DEVICE)
   
    if train:
        logger.info("START TRAINING")
        df_tubes = pd.read_csv(f"{data_path}/processed_labels.csv")
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion.requires_grad = True
        optimizer = optim.Adam(model.parameters(),lr=lr)
        
        df_tubes_annotations = df_tubes[df_tubes["Drain"]!=-1]
        train_test_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1907)
        for train_idx, test_idx in train_test_split.split(df_tubes_annotations, groups=df_tubes_annotations['Patient ID']):
            df_train = df_tubes_annotations.iloc[train_idx]
            df_test = df_tubes_annotations.iloc[test_idx]
            training_data = TubesDataset(labels=df_train,data_path=data_path)
            testing_data = TubesDataset(labels=df_test,data_path=data_path)
            
            train_dataloader = DataLoader(training_data, batch_size=batch_size)
            test_dataloader = DataLoader(testing_data, batch_size=batch_size)
            with radt.run.RADTBenchmark() as run:
                run.log_param("nb_epochs", nb_epochs)
                run.log_param("lr", lr)
                run.log_param("batch_size", batch_size)
                for epoch in tqdm(range(nb_epochs)):
                    train_loss,train_metric = training_epoch(model,criterion,optimizer,train_dataloader)
                    print("Train metrics:",train_loss,train_metric)
                    run.log_metric("train_loss", train_loss,epoch)
                    run.log_metric("train_auc", train_metric,epoch)
                    torch.save(model.state_dict(),f'{MODELS_DIR}/drains_detection_model.pt')
                test_auc = test_model(model,test_dataloader)
                run.log_metric("test_auc", test_auc,epoch)
                logger.info("Test AUC:",test_auc)
        logger.success("TRAINING COMPLETED")
    else:
        #Apply to PadChest data
        logger.info("START DRAINS DETECION")

        logger.info("For train split")
        lst_train_datasets = {dataset.dataset_name:dataset for dataset in get_padchest_datasets_to_evaluate()}
        padchest_train_dataset = lst_train_datasets.get("All")
        padchest_train_loader = DataLoader(padchest_train_dataset, batch_size=32)
        drains_preds = get_padchest_preds(model,padchest_train_loader)
        padchest_train_dataset.labels_csv["drains_proba"] = drains_preds.flatten()
        padchest_train_dataset.labels_csv.to_csv(f"{PROCESSED_DATA_DIR}/padchest/train_labels_drains.csv",sep=",")

        logger.info("For test split")
        padchest_test_dataset = get_padchest_test()
        padchest_test_loader = DataLoader(padchest_test_dataset, batch_size=32)
        drains_preds = get_padchest_preds(model,padchest_test_loader)
        padchest_test_dataset.labels_csv["drains_proba"] = drains_preds.flatten()
        padchest_train_dataset.labels_csv.to_csv(f"{PROCESSED_DATA_DIR}/padchest/test_labels_drains.csv",sep=",")
        logger.success("DRAINS DETECION COMPLETED")

if __name__ == "__main__":
    typer.run(main)