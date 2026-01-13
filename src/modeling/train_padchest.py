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
from torch.nn.functional import sigmoid
from sklearn.metrics import roc_auc_score

import radt
import mlflow

from src.data.padchest_pytorch import get_padchest_datasets_to_evaluate,get_padchest_test


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
        inputs, texts, labels, img_ids, dataset_names = data
        inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        output_sigmoid = sigmoid(outputs).flatten()
        loss = criterion(output_sigmoid, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        lst_labels.extend(labels.cpu().detach().numpy())
        lst_probas.extend(output_sigmoid.cpu().detach().numpy())

    lst_labels = np.array(lst_labels)
    lst_probas = np.array(lst_probas)
    auc_scores=roc_auc_score(lst_labels,lst_probas)
    print(f"train ({len(lst_labels)} images)",auc_scores,flush=True)

    return train_loss,auc_scores

def valid_epoch(model,criterion,dataloader):
    model.to(DEVICE)
    model.eval()

    val_loss = 0.0
    lst_labels = []
    lst_probas = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, texts, labels, img_ids, dataset_names = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)

            # forward + backward
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs).flatten()
            loss = criterion(output_sigmoid, labels)
            val_loss += loss.item()
            
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())

        lst_labels = np.array(lst_labels)
        lst_probas = np.array(lst_probas)
        auc_scores=roc_auc_score(lst_labels,lst_probas,average=None,multi_class='ovr')
        print(f"val ({len(lst_labels)} images)",auc_scores,flush=True)

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
            inputs, texts, labels, img_ids, dataset_names = data
            inputs,labels = inputs.float().to(DEVICE), torch.Tensor(labels).float().to(DEVICE)
            
            # forward + backward
            outputs = model(inputs)
            output_sigmoid = sigmoid(outputs).flatten()
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_sigmoid.cpu().detach().numpy())
            lst_img_ids.extend(img_ids)
            lst_dataset_names.extend(dataset_names)
    return lst_img_ids,lst_dataset_names,lst_labels,lst_probas

@app.command()
def main(
    dataset_config: str = "All",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 5,
    checkpoint_path: Path = None
):
    lst_train_datasets = {dataset.dataset_name:dataset for dataset in get_padchest_datasets_to_evaluate()}
    
    train_dataset = lst_train_datasets.get(dataset_config)
    if not train_dataset:
            logger.error(f"Config name '{dataset_config}' not matching any possible configs. List of possible configs:{lst_train_datasets.keys()}")
            return    
    test_dataset = get_padchest_test()
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    with radt.run.RADTBenchmark() as run:
        
        param_message = f"TRANING INFO\nDATASET CONFIG: {dataset_config}\n"
        param_message += f"NUMBER OF EPOCHS: {epochs}\n"
        param_message += f"Learning rate: {lr}\n"
        param_message += f"BATCH SIZE: {batch_size}"


        run.log_param("dataset_config", dataset_config)
        # run.log_param("max_epoch", epochs)
        run.log_param("lr", lr)
        run.log_param("batch_size", batch_size)

        training_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        training_folder = INTERIM_DATA_DIR / f"padchest_trainings/{dataset_config}_{training_datetime}"
        Path(training_folder).mkdir(parents=True, exist_ok=True)
        with open(training_folder/"parameters","w") as metrics_csvfile:
            metrics_csvfile.write(param_message)
        logger.info(param_message)


        skf = StratifiedKFold(n_splits=5)

        X = train_dataset.labels_csv
        y = train_dataset.labels_csv["label"]
        
        for i, (train_index, val_index) in enumerate(skf.split(X, y)):
            with mlflow.start_run(run_name=f'fold_{i}', nested=True):
                mlflow.log_param("fold", i)
                logger.info(f"Starting fold {i}")
                #Create the training and validation splits

                train_fold = deepcopy(train_dataset)
                train_fold.labels_csv = train_fold.labels_csv.iloc[train_index].reset_index(drop=True)
                
                val_fold = deepcopy(train_dataset)
                val_fold.labels_csv = val_fold.labels_csv.iloc[val_index].reset_index(drop=True)
                
                #Create the Dataloaders
                train_dataloader = DataLoader(train_fold, batch_size=batch_size)
                valid_dataloader = DataLoader(val_fold, batch_size=batch_size)
                
                #Instanciate the model, criterion and optimizer
                model = resnet50()
                model.fc = torch.nn.Linear(model.fc.in_features, 1)

                criterion = torch.nn.BCELoss()
                optimizer = optim.Adam(model.parameters(),lr=lr)

                datamaps_info = {}
                best_val_loss = np.inf
                best_epoch = 0
                #Training/Evaluation loop
                for e in tqdm(range(epochs)):
                    run.log_metric("epoch", e)
                    train_loss,auc_train= training_epoch(model,criterion,optimizer,train_dataloader)
                    val_loss,auc_val = valid_epoch(model,criterion,valid_dataloader)

                    #Compute pred on test set for datamap
                    lst_img_ids,lst_dataset_names,lst_labels,lst_probas = compute_datamap_info(model,test_dataloader)
                    for j in range(len(lst_img_ids)):
                        sample_id = lst_img_ids[j]
                        if sample_id not in datamaps_info:
                            datamaps_info[sample_id] = {
                                "img_id":lst_img_ids[j],
                                "dataset_name":lst_dataset_names[j],
                                "label":lst_labels[j],
                                "proba_label":[float(lst_probas[j])]
                            }
                        else:
                            datamaps_info[sample_id]["proba_label"].append(float(lst_probas[j]))
                    
                    pd.DataFrame.from_dict(datamaps_info, orient='index').to_csv(training_folder/f"datamaps_values_fold{i}.csv")

                    with open(training_folder/f"training_loss_fold{i}.csv","a") as metrics_csvfile:
                        metrics_csvfile.write(f"{train_loss},{val_loss}\n")
                    mlflow.log_metric("train_loss", train_loss)
                    mlflow.log_metric("val_loss", val_loss)
                    print(train_loss,val_loss)

                    if val_loss < best_val_loss:
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict(),
                            'last_epoch':e
                        }, training_folder/f"checkpoint_fold{i}.pth")
                        best_val_loss = val_loss
                        best_epoch = e
                    else:
                        if e - best_epoch > patience:
                            logger.info("Early stopped, best epoch:", best_epoch)
                            break

                logger.success(f"Completed fold {i}")
    
if __name__ == "__main__":
    app()
