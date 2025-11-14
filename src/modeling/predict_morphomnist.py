from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import v2
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from src.data.morphomnist_pytorch import get_perturb_dataset,get_test_dataset
from glob import glob

torch.manual_seed(1907)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer()


def compute_preds(model,dataloader):
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
            
            # forward
            outputs = model(inputs)
            output_softmax = softmax(outputs,dim=1)
            lst_labels.extend(labels.cpu().detach().numpy())
            lst_probas.extend(output_softmax.cpu().detach().numpy())
            lst_img_ids.extend(img_ids)
            lst_dataset_names.extend(dataset_names)

    return lst_img_ids,lst_dataset_names,lst_labels,lst_probas

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    models_folders: Path = INTERIM_DATA_DIR/"morphomnist_trainings",
    # -----------------------------------------
):
    test_dataset = get_test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    with open(PROCESSED_DATA_DIR/f"morphomnist_aucs.csv","w") as perf_file:
        perf_file.write("model,mean_auc,std_auc")

    #iterate through the folders in morphomnist_trainings excluding .DS_STORE
    training_folders = list(models_folders.glob("[a-z]*"))
    for folder in training_folders:
        #get the model name from the folder path removimg the training timestamp (last 15 characters)
        model_name = folder.parts[-1][:-15]
        logger.info(f"Compute prediction for {model_name}")
        folds_results = []
        for fold_nb in tqdm(range(5)):
            checkpoint_path = f"{folder}/checkpoint_fold{fold_nb}.pth"
            
            #Instanciate the model and load weights
            model = resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            checkpoint = torch.load(checkpoint_path,map_location=DEVICE)
            model.load_state_dict(checkpoint['model'])

            lst_img_ids,lst_dataset_names,lst_labels,lst_probas = compute_preds(model,test_dataloader)
            preds_csv = {}
            for j in range(len(lst_img_ids)):
                sample_id = f"{lst_img_ids[j]}_{lst_dataset_names[j]}"
                preds_csv[sample_id] = {
                    "img_id":lst_img_ids[j],
                    "dataset_name":lst_dataset_names[j],
                    "label":lst_labels[j],
                    "proba_label":lst_probas[j]
                }
            folds_results.append(roc_auc_score(lst_labels,lst_probas,multi_class="ovr"))
            pd.DataFrame.from_dict(preds_csv, orient='index').to_csv(f"{folder}/predictions_fold{fold_nb}.csv")
         

        with open(PROCESSED_DATA_DIR/f"morphomnist_aucs.csv","a+") as perf_file:
            perf_file.write(f"\n{model_name},{np.mean(folds_results)},{np.std(folds_results)}")


if __name__ == "__main__":
    app()
