from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset
from copy import deepcopy

from torchvision.models import resnet50
from torchvision.transforms import v2
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from src.data.morphomnist_pytorch import get_perturb_dataset,get_test_dataset


from src.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

torch.manual_seed(1907)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_epoch(model,criterion,optimizer,train_dataloader):
    model.to(DEVICE)
    model.train()

    train_loss = 0.0
    lst_labels = []
    lst_probas = []
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels, img_ids, dataset_names = data
        inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        output_softmax = softmax(outputs,dim=1)
        loss = criterion(output_softmax, labels.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        lst_labels.extend(labels.cpu().detach().numpy())
        lst_probas.extend(output_softmax.cpu().detach().numpy())

    lst_labels = np.array(lst_labels)
    lst_probas = np.array(lst_probas)
    auc_scores=roc_auc_score(lst_labels,lst_probas,average=None,multi_class='ovr')
    # print(f"train ({len(lst_labels)} images)",auc_scores,flush=True)

    return train_loss,auc_scores

def valid_epoch(model,criterion,dataloader):
    model.to(DEVICE)
    model.eval()

    val_loss = 0.0
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, img_ids, dataset_names = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)

            # forward + backward
            outputs = model(inputs)
            output_softmax = softmax(outputs,dim=1)
            loss = criterion(output_softmax, labels.long())
            val_loss += loss.item()
            
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_softmax.cpu().detach().numpy())

        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)
        auc_scores=roc_auc_score(lst_labels,lst_probas,average=None,multi_class='ovr')
        # print(f"val ({len(lst_labels)} images)",auc_scores,flush=True)

    return val_loss,auc_scores

def compute_datamap_info(model,dataloader):
    model.to(DEVICE)
    model.eval()
    
    lst_labels = []
    lst_probas = []
    lst_img_ids = []
    lst_dataset_names = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, img_ids, dataset_names = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)
            
            # forward + backward
            outputs = model(inputs)
            output_softmax = softmax(outputs,dim=1)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_softmax.cpu().detach().numpy())
            lst_img_ids.extend(img_ids)
            lst_dataset_names.extend(dataset_names)
    return lst_img_ids,lst_dataset_names,lst_labels,lst_probas

@app.command()
def main(
    dataset_config: str = "plain",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    checkpoint_path: Path = None
):
    lst_train_datasets = {dataset.dataset_name:dataset for dataset in get_perturb_dataset()}
    
    test_dataset = get_test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    for ds_name in lst_train_datasets:
        train_dataset = lst_train_datasets.get(ds_name)
        if not train_dataset:
            logger.error(f"Config name '{ds_name}' not matching any possible configs. List of possible configs:{lst_train_datasets.keys()}")
            return
        

        param_message = f"TRANING INFO\nDATASET CONFIG: {ds_name}\n"
        param_message += f"NUMBER OF EPOCHS: {epochs}\n"
        param_message += f"Learning rate: {lr}\n"
        param_message += f"BATCH SIZE: {batch_size}"

        training_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        training_folder = INTERIM_DATA_DIR / f"morphomnist_trainings/{ds_name}_{training_datetime}"
        Path(training_folder).mkdir(parents=True, exist_ok=True)
        with open(training_folder/"parameters","w") as metrics_csvfile:
            metrics_csvfile.write(param_message)
        logger.info(param_message)


        skf = StratifiedKFold(n_splits=5)
        if isinstance(train_dataset,ConcatDataset):
            X = train_dataset.datasets[0].labels_csv
            y = train_dataset.datasets[0].labels_csv["label"]
        else:
            X = train_dataset.labels_csv
            y = train_dataset.labels_csv["label"]
        
        for i, (train_index, val_index) in enumerate(skf.split(X, y)):
            logger.info(f"Starting fold {i}")
            #Create the training and validation splits
            #In case of combination of datasets (ConcatDataset), we compute the train and val split indexes on the first dataset and take the same indexes
            #for the other datasets to keep the same "base" digit and avoid data leakage
            if isinstance(train_dataset,ConcatDataset):
                lst_datasets_train = []
                lst_datasets_val = []
                for _,d in enumerate(train_dataset.datasets):
                    #Copy the dataset, filter to keep training samples and add to the list used later to create the ConcatDataset
                    d_copy_train = deepcopy(d)
                    d_copy_train.labels_csv = d_copy_train.labels_csv.iloc[train_index].reset_index(drop=True)
                    d_copy_train.imgs = d_copy_train.imgs[train_index]
                    lst_datasets_train.append(d_copy_train)
                    
                    #Similar to training samples but for the validation split
                    d_copy_val = deepcopy(d)
                    d_copy_val.labels_csv = d_copy_val.labels_csv.iloc[val_index].reset_index(drop=True)
                    d_copy_val.imgs = d_copy_val.imgs[val_index]
                    lst_datasets_val.append(d_copy_val)
                
                #Create the ConcatDataset from the list of filtered datasets
                train_fold = ConcatDataset(lst_datasets_train)
                train_fold.as_tensor = train_dataset.datasets[0].as_tensor
                
                val_fold = ConcatDataset(lst_datasets_val)
                val_fold.as_tensor = train_dataset.datasets[0].as_tensor
            else:
                #Case of single dataset, filter the data to keep the training samples only
                train_fold = deepcopy(train_dataset)
                train_fold.labels_csv = train_fold.labels_csv.iloc[train_index].reset_index(drop=True)
                train_fold.imgs = train_fold.imgs[train_index]
                
                #Same for validation split
                val_fold = deepcopy(train_dataset)
                val_fold.labels_csv = val_fold.labels_csv.iloc[val_index].reset_index(drop=True)
                val_fold.imgs = val_fold.imgs[val_index]
            
            #Create the Dataloaders
            train_dataloader = DataLoader(train_fold, batch_size=batch_size)
            valid_dataloader = DataLoader(val_fold, batch_size=batch_size)
            
            #Instanciate the model, criterion and optimizer
            model = resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),lr=lr)

            datamaps_info = {}
            #Training/Evaluation loop
            for e in tqdm(range(epochs)):
                train_loss,aucs_val = training_epoch(model,criterion,optimizer,train_dataloader)
                val_loss,aucs_val = valid_epoch(model,criterion,valid_dataloader)

                #Compute pred on test set for datamap
                lst_img_ids,lst_dataset_names,lst_labels,lst_probas = compute_datamap_info(model,test_dataloader)
                for j in range(len(lst_img_ids)):
                    sample_id = f"{lst_img_ids[j]}_{lst_dataset_names[j]}"
                    if sample_id not in datamaps_info:
                        datamaps_info[sample_id] = {
                            "img_id":lst_img_ids[j],
                            "dataset_name":lst_dataset_names[j],
                            "label":lst_labels[j],
                            "proba_label":[float(lst_probas[j][int(lst_labels[j])])]
                        }
                    else:
                        datamaps_info[sample_id]["proba_label"].append(float(lst_probas[j][int(lst_labels[j])]))
                
                pd.DataFrame.from_dict(datamaps_info, orient='index').to_csv(training_folder/f"datamaps_values_fold{i}.csv")

                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                    'last_epoch':e
                }, training_folder/f"checkpoint_fold{i}.pth")

                with open(training_folder/f"training_loss_fold{i}.csv","a") as metrics_csvfile:
                    metrics_csvfile.write(f"{train_loss},{val_loss}\n")
                print(train_loss,val_loss)

            logger.success(f"Completed fold {i}")
    
if __name__ == "__main__":
    app()
