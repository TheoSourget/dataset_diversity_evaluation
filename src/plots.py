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
import math

from src.config import REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

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

base_params_plots = {
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
mpl.rcParams.update(base_params_plots)

app = typer.Typer()

#FIGURES
def metrics_ranking_correlation_matrix(metrics_path,auc_path):
    metric_name_to_table = {
        "inception_score":"IS",
        "fid":"FID",
        "vs_pixel":"VS_pix",
        "vs_hog":"VS_hog",
        "vs_inception":"VS_Inception",
        "rougeL":"RougeL",
        "semantic_similarity":"Semantic",
        "metadata_similiarity":"Metadata",
        "AUC":"AUC",
    }
    params_plots = {
        "axes.grid": False,
        "axes.facecolor": "#ffffff",
    }
    mpl.rcParams.update(params_plots)
    ascending = [False,True,False,False,False,True,True,True,False]

    #Load the csv with metrics values
    csv_metrics = pd.read_csv(metrics_path,index_col="metric_name")
    csv_metrics = csv_metrics.map(lambda x: float(x.split("_")[0]))
    
    auc_df = pd.read_csv(auc_path,index_col="model")
    csv_metrics.loc["AUC"] = auc_df["mean_auc"]
    
    #For each row compute the ranking
    rankings = []
    for i,r in enumerate(csv_metrics.iterrows()):
        rankings.append(r[1].rank(ascending=ascending[i]).astype(int).to_numpy())
    
    # rankings = csv_metrics.rank(axis = 1, ).astype(int).to_numpy()
    rankings = np.array(rankings)
    fig = plt.figure()
    sns.heatmap(rankings,annot=True,vmin=1,vmax=5,xticklabels=csv_metrics.columns,yticklabels=metric_name_to_table.values(),cmap="coolwarm")
    plt.title(f"Ranking of datasets per metric")
    plt.tight_layout()

    #Save the figure
    fig.savefig(FIGURES_DIR / f"metrics_ranking_values.png")

    #Compute the correlation per pair of ranking
    corrs = stats.spearmanr(rankings,axis=1).statistic
    mask_matrix = np.tril(np.ones_like(corrs))

    #Display the correlations in a matrix format
    fig = plt.figure()
    sns.heatmap(corrs,annot=True,vmin=-1,vmax=1,xticklabels=metric_name_to_table.values(),yticklabels=metric_name_to_table.values(),cmap="BrBG",mask=mask_matrix,)
    plt.title(f"Correlation between the ranking of the metrics")
    plt.tight_layout()

    #Save the figure
    fig.savefig(FIGURES_DIR / f"metrics_ranking_correlation.png")
    mpl.rcParams.update(base_params_plots)
    

def evolution_metrics(csv_path,parameter_name):
    metric_name_to_table = {
        "inception_score":"IS +",
        "fid":"FID -",
        "vs_pixel":"Vendi Score (pixel values) +",
        "vs_hog":"Vendi Score (HoG) +",
        "vs_inception":"Vendi Score (Inception) +"
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
        plt.ylabel(metric_name_to_table.get(metric_name,metric_name))

        plt.xticks(x)
        plt.title(f"Evolution of the {metric_name} when varying the {parameter_name} parameter",fontsize=15)
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
    
    #Get the probabilities across training epochs
    df_pred = pd.read_csv(pred_path)
    df_pred["proba_label"] = df_pred["proba_label"].apply(lambda x: np.array(ast.literal_eval(x)))
    preds = np.array(df_pred["proba_label"].to_list())[:,:-6] #Remove end of the array made of the early stopping patience

    #Compute confidence (mean probability) and variability (standard deviation) for all samples
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
    plt.xlabel(f"Variability")
    plt.ylabel(f"Confidence")
    plt.title(caption)
    plt.tight_layout()

    path_figures = FIGURES_DIR/f"datamaps/{model_name}"
    path_figures.mkdir(parents=True, exist_ok=True)

    fig.savefig(path_figures/f"datamap_{model_name}_model.png",dpi=500)

    for ds_perturbation in grp_id:
        df_ds = df_pred[df_pred["dataset_name"]==ds_perturbation]
        preds = np.array(df_ds["proba_label"].to_list())[:,:-6] #Remove end of the array made of the early stopping patience
        confidence = np.mean(preds,axis=1)
        variability = np.std(preds,axis=1)
        nbins = 100
        k = kde(np.vstack([variability, confidence]))
        xi, yi = np.mgrid[0:1:nbins*1j, 0:1:nbins*1j]

        #Use the square root for a better color range 
        zi = np.sqrt(k(np.vstack([xi.flatten(), yi.flatten()])))
        fig = plt.figure()
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud',cmap=cm.Blues)
        plt.title(f"{ds_perturbation} images",fontsize=20)
        plt.axis('square')
        plt.xlabel("Variability",fontsize=20)
        plt.ylabel("Confidence",fontsize=20)
        plt.tight_layout()
        save_path = path_figures/f"density_{ds_perturbation}__{model_name}_model.png"
        fig.savefig(save_path,dpi=500,bbox_inches='tight')
        plt.close()


#TABLES
def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

def generate_table_metrics(metric_file,auc_file,output_file):
    metric_name_to_table = {
        "inception_score":"IS ↑",
        "fid":"FID ↓",
        "vs_pixel":"Vendi Score (pixel values) ↑",
        "vs_hog":"Vendi Score (HoG) ↑",
        "vs_inception":"Vendi Score (Inception) ↑",
        "rougeL":"RougeL ↓",
        "semantic_similarity":"Semantic diversity ↓",
        "metadata_similarity":"Metadata diversity ↓"
    }
    metrics_df = pd.read_csv(metric_file,index_col="metric_name").T
    auc_df = pd.read_csv(auc_file,index_col="model")

    with open(output_file, "w") as text_file:
        #Table definition, caption, and label
        text_file.write("\\begin{table}[tb]\n")
        text_file.write("\\centering\n")
        text_file.write("\\caption{Metrics on the Morpho-MNIST dataset. + indicates that a higher value is best, - indicates that a lower value is best. Values in [] indicate the 95\% confidence interval computed with the bootstrap method. +/- for AUC scores indicate the standard deviation of the five models trained using 5-fold cross-validation on the test set.}\n")
        text_file.write("\\label{tab:morphomnist_metrics}\n")
        text_file.write("\\resizebox{\\textwidth}{!}{%\n")
        text_file.write("\\begin{tabular}{l|l|l|l|l|l|l|l|l|l|}\n")
        
        #Column header
        text_file.write("\t\t\t\t & Plain & Thin & Thick & Fracture & Swelling & Plain $\cup$ Thin & Plain $\cup$ Thick & Plain $\cup$ Fracture & Plain $\cup$ Swelling"+r" \\"+"\n")
        text_file.write("\\hline\n")
        
        #Diversity metrics
        for metric in metrics_df:
            text_file.write(f"{metric_name_to_table[metric]}\t")
            for dataset_value in metrics_df[metric]:
                text_file.write(f"&{truncate(float(dataset_value.split('_')[0]),2)}\t")
            text_file.write(r"\\"+"\n")
            for dataset_value in metrics_df[metric]:
                if metric == "rougeL":
                    text_file.write(f"&$\pm${truncate(float(dataset_value.split('_')[1]),4)}\t")
                else:
                    text_file.write(f"&[{truncate(float(dataset_value.split('_')[1]),2)},{truncate(float(dataset_value.split('_')[2]),2)}]\t")
            text_file.write(r"\\"+"\n")
        text_file.write("\\hline\n")

        #AUC
        text_file.write("AUC ↑\t")
        for auc_dataset in auc_df["mean_auc"]:
            text_file.write(f"&{truncate(float(auc_dataset),3)}\t")
        text_file.write(r"\\"+"\n")
        for std_dataset in auc_df["std_auc"]:
            text_file.write(f"&$\pm${truncate(float(std_dataset),3)}\t")
        text_file.write(r"\\"+"\n") 
        text_file.write("\\hline\n")

        text_file.write("\\end{tabular}\n")
        text_file.write("}\n")
        text_file.write("\\end{table}\n")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    metrics_csv_path = INTERIM_DATA_DIR / "diversity_metrics_morphomnist.csv"

    logger.info("Generating plot from data...")
    # metrics_ranking_correlation_matrix(INTERIM_DATA_DIR/"diversity_metrics_padchest.csv",PROCESSED_DATA_DIR/"padchest_aucs.csv")
    # metrics_values_matrix(metrics_csv_path)
    # evolution_metrics(INTERIM_DATA_DIR / "thinning_diversity_metrics.csv","thinning")
    # evolution_metrics(INTERIM_DATA_DIR / "thickening_diversity_metrics.csv","thickening")
    # generate_table_metrics(metrics_csv_path,PROCESSED_DATA_DIR/"morphomnist_aucs.csv",REPORTS_DIR/"padchest_diversity_metrics_value.tex")
    # generate_table_metrics(metrics_csv_path,PROCESSED_DATA_DIR/"padchest_aucs.csv",REPORTS_DIR/"padchest_diversity_metrics_value.tex")

    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_20260218125322/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_model.png","Datamap of model trained on plain images","plain")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/thin_20260218141219/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/thin_model.png","Datamap of model trained on thin images","thin")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/thick_20260218152922/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/thick_model.png","Datamap of model trained on thick images","thick")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/swelling_20260218162205/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/swelling_model.png","Datamap of model trained on swelling images","swelling")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/fracture_20260218175236/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/fracture_model.png","Datamap of model trained on fracture images","fracture")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_thin_20260218191852/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_thin_model.png","Datamap of model trained on plain and thin images","plain_thin")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_thick_20260218205323/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_thick_model.png","Datamap of model trained on plain and thick images","plain_thick")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_swelling_20260218221853/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_swelling_model.png","Datamap of model trained on plain and swelling images","plain_swelling")
    datamap(INTERIM_DATA_DIR / "morphomnist_trainings/plain_fracture_20260219002322/datamaps_values_fold0.csv",FIGURES_DIR/"datamaps/plain_fracture_model.png","Datamap of model trained on plain and fracture images","plain_fracture")

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
