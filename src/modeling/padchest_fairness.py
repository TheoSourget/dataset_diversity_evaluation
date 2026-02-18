from pathlib import Path
import typer
from sklearn.metrics import roc_auc_score
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
import numpy as np
import pandas as pd
from src.data.padchest_pytorch import get_padchest_datasets_to_evaluate,get_padchest_test
app = typer.Typer()

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    models_folders: Path = INTERIM_DATA_DIR/"padchest_trainings",
    # -----------------------------------------
):
    test_dataset = get_padchest_test()
    df_metadata = test_dataset.labels_csv

    training_folders = list(models_folders.glob("[A-Z][a-z]*"))

    with open(PROCESSED_DATA_DIR/f"padchest_fairness_aucs.csv","w") as perf_file:
        perf_file.write("model,mean_auc_male,std_auc_male,mean_auc_female,std_auc_female,mean_auc_philips,std_auc_philips,mean_auc_imagingdynamics,std_auc_imagingdynamics")

    for folder in training_folders:
        model_name = folder.parts[-1][:-15]
        male_aucs = []
        female_aucs = []
        philips_aucs = []
        imagingdynamics_aucs = []

        for fold_nb in range(5):
            df_predictions = pd.read_csv(f"{folder}/predictions_fold{fold_nb}.csv")
            df_predictions["PatientSex_DICOM"] = df_metadata["PatientSex_DICOM"]
            df_predictions["Manufacturer_DICOM"] = df_metadata["Manufacturer_DICOM"]

            #Computing AUC per subgroup
            #Male
            df_predictions_male = df_predictions[df_predictions["PatientSex_DICOM"]=="M"]
            male_aucs.append(roc_auc_score(df_predictions_male["label"],df_predictions_male["proba_label"]))
            
            # Female
            df_predictions_female = df_predictions[df_predictions["PatientSex_DICOM"]=="F"]
            female_aucs.append(roc_auc_score(df_predictions_female["label"],df_predictions_female["proba_label"]))

            # Philips scanner
            df_predictions_philips = df_predictions[df_predictions["Manufacturer_DICOM"]=="PhilipsMedicalSystems"]
            philips_aucs.append(roc_auc_score(df_predictions_philips["label"],df_predictions_philips["proba_label"]))

            # ImagingDynamics
            df_predictions_imagingdynamics = df_predictions[df_predictions["Manufacturer_DICOM"]=="ImagingDynamicsCompanyLtd"]
            imagingdynamics_aucs.append(roc_auc_score(df_predictions_imagingdynamics["label"],df_predictions_imagingdynamics["proba_label"]))
        with open(PROCESSED_DATA_DIR/f"padchest_fairness_aucs.csv","a+") as perf_file:
            perf_file.write(f"\n{model_name},{np.mean(male_aucs)},{np.mean(male_aucs)}")
            perf_file.write(f",{np.mean(female_aucs)},{np.std(female_aucs)}")
            perf_file.write(f",{np.mean(philips_aucs)},{np.std(philips_aucs)}")
            perf_file.write(f",{np.mean(imagingdynamics_aucs)},{np.std(imagingdynamics_aucs)}")

if __name__ == "__main__":
    app()
