from pathlib import Path
from turtle import color

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import gaussian_kde as kde
import ast
import numpy as np

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

import matplotlib.colors as mcolors
import matplotlib as mpl
class UCBerkeley:
    """From https://brand.berkeley.edu/colors/"""

    info = [
        {"hex_value": "#002676", "name": "Berkeley blue", "type": "primary"},
        {"hex_value": "#018943", "name": "Green Medium", "type": "primary"},
        {"hex_value": "#FDB515", "name": "California Gold", "type": "primary"},
        {"hex_value": "#E7115E", "name": "Rose Medium", "type": "primary"},
        {"hex_value": "#6C3302", "name": "South Hall", "type": "primary"},
        {"hex_value": "#FF0000", "name": "Red", "type": "primary"},
    ]
    colors = [d["hex_value"] for d in info]

style_params = {
    "axes.grid": True,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.facecolor": "#ebebeb",
    "axes.axisbelow": True,
    "axes.titlelocation": "center",
    "grid.color": "white",
    "grid.linestyle": "-",
    "grid.linewidth": 1,
    "grid.alpha": 1,
    "xtick.color": "#4D4D4D",
    "ytick.color": "#4D4D4D",
    "text.color": "#000000",
    "font.family": ["arial"],
    "image.cmap": "viridis",
    "axes.prop_cycle": mpl.cycler(color=UCBerkeley.colors),
}
mpl.rcParams.update(style_params)

app = typer.Typer()

def metrics_ranking_correlation_matrix(csv_path):
    #Load the csv with metrics values
    csv_metrics = pd.read_csv(csv_path,index_col="metric_name")
    csv_metrics = csv_metrics.map(lambda x: float(x.split("_"))[0])
    #For each row compute the ranking
    rankings = csv_metrics.rank(axis = 1, ascending = [False,True,False,False,False]).astype(int).to_numpy()
    #Compute the correlation per pair of ranking
    corrs = stats.spearmanr(rankings,axis=1).statistic
    #Display the correlations in a matrix format
    fig = plt.figure()
    sns.heatmap(corrs,annot=True,vmin=-1,vmax=1,xticklabels=csv_metrics.index.values,yticklabels=csv_metrics.index.values)
    plt.title(f"Correlation between the ranking of the metrics")
    plt.tight_layout()

    #Save the figure
    fig.savefig(FIGURES_DIR / f"metrics_ranking_correlation.png")
   

def metrics_values_matrix(csv_path):
    #Load the csv with metrics values
    csv_metrics = pd.read_csv(csv_path,index_col="metric_name")

    #Plot the metrics as a matrix format
    fig = plt.figure()
    sns.heatmap(csv_metrics,annot=True,vmin=0,vmax=1)
    plt.title(f"Metrics value for different dataset versions")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"metrics_values.png")
    

def evolution_metrics(csv_path,parameter_name):
    metrics_label = {
        "vs":"Vendi Score +",
        "inception_score":"Inception score +",
        "fid":"Fréchet Inception distance -"
    }
    #Load the csv with metrics values
    csv_metrics = pd.read_csv(csv_path,index_col="metric_name").T
    path_figures = FIGURES_DIR / f"evolution/{parameter_name}"
    path_figures.mkdir(parents=True, exist_ok=True)
    for metric_name in csv_metrics.columns:
        fig, ax = plt.subplots()
        
        x=[int(param.split("_")[1]) for param in csv_metrics.index]

        metric_values =  csv_metrics[metric_name].apply(lambda x: float(x.split("_")[0]))
        low_CI_values =  csv_metrics[metric_name].apply(lambda x: float(x.split("_")[1]))
        up_CI_values =  csv_metrics[metric_name].apply(lambda x: float(x.split("_")[2]))

        ax.plot(x,metric_values,marker="o")
        ax.fill_between(x, low_CI_values, up_CI_values, alpha=.1)

        plt.xlabel(f"{parameter_name} parameter")
        plt.ylabel(metrics_label.get(metric_name,metric_name))

        plt.xticks(x)
        plt.title(f"Evolution of the {metric_name} when varying the {parameter_name} parameter")
        fig.tight_layout()
        fig.savefig(path_figures / f"{metric_name}.png")

def datamap(pred_path,save_path,caption,model_name):
    grp_id = {
        "plain": 0,
        "thin": 1,
        "thick": 2,
        "swelling": 3,
        "fracture": 4,
    }
    df_pred = pd.read_csv(pred_path)
    df_pred["proba_label"] = df_pred["proba_label"].apply(lambda x: np.array(ast.literal_eval(x)))
    preds = np.array(df_pred["proba_label"].to_list())
    confidence = np.mean(preds,axis=1)
    variability = np.std(preds,axis=1)
    fig = sns.jointplot(x=variability, y=confidence, hue=df_pred["dataset_name"], kind='scatter', alpha=0.6, marker='o', s=40, hue_order=grp_id.keys(), joint_kws=dict(rasterized=True))
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = plt.annotate("ambiguous", xy=(0.9, 0.5), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", bbox=bb('black'))
    an2 = plt.annotate("easy-to-learn", xy=(0.27, 0.85), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", bbox=bb('r'))
    an3 = plt.annotate("hard-to-learn", xy=(0.4, 0.1), xycoords="axes fraction", fontsize=15, color='black',
                  va="center", ha="center", bbox=bb('b'))
    fig.ax_joint.legend()
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel(f"variability")
    plt.ylabel(f"confidence")
    plt.title(caption)
    plt.tight_layout()

    path_figures = FIGURES_DIR/f"datamaps/{model_name}"
    path_figures.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_figures/f"datamap_{model_name}_model.png",dpi=500)

    for ds_perturbation in grp_id:
        df_ds = df_pred[df_pred["dataset_name"]==ds_perturbation]
        preds = np.array(df_ds["proba_label"].to_list())
        confidence = np.mean(preds,axis=1)
        variability = np.std(preds,axis=1)
        nbins = 100
        k = kde(np.vstack([variability, confidence]))
        xi, yi = np.mgrid[0:1:nbins*1j, 0:1:nbins*1j]

        #Use the square root to 
        zi = np.sqrt(k(np.vstack([xi.flatten(), yi.flatten()])))
        fig = plt.figure()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud',cmap=cm.Blues)
        plt.title(f"Density plot of datamap for {ds_perturbation} images with model trained on {model_name.replace('_',' and ')} images")
        plt.axis('square')
        plt.tight_layout()
        save_path = path_figures/f"density_{ds_perturbation}__{model_name}_model.png"
        fig.savefig(save_path,dpi=500)
        plt.close()
    
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    metrics_csv_path = INTERIM_DATA_DIR / "diversity_metrics.csv"

    logger.info("Generating plot from data...")
    # metrics_ranking_correlation_matrix(metrics_csv_path)
    # metrics_values_matrix(metrics_csv_path)
    # evolution_metrics(INTERIM_DATA_DIR / "thinning_diversity_metrics.csv","thinning")
    # evolution_metrics(INTERIM_DATA_DIR / "thickening_diversity_metrics.csv","thickening")

    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_20251107124546/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_model.png","Datamap of model trained on plain images","plain")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/thin_20251107202007/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/thin_model.png","Datamap of model trained on thin images","thin")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/thick_20251108034859/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/thick_model.png","Datamap of model trained on thick images","thick")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/swelling_20251108112330/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/swelling_model.png","Datamap of model trained on swelling images","swelling")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/fracture_20251108185345/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/fracture_model.png","Datamap of model trained on fracture images","fracture")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_thin_20251109022338/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_thin_model.png","Datamap of model trained on plain and thin images","plain_thin")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_thick_20251109150701/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_thick_model.png","Datamap of model trained on plain and thick images","plain_thick")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_swelling_20251110034309/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_swelling_model.png","Datamap of model trained on plain and swelling images","plain_swelling")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_fracture_20251110163053/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_fracture_model.png","Datamap of model trained on plain and fracture images","plain_fracture")

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
