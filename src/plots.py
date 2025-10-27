from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    metrics_csv_path = INTERIM_DATA_DIR / "diversity_metrics.csv"

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    metrics_ranking_correlation_matrix(metrics_csv_path)
    metrics_values_matrix(metrics_csv_path)
    evolution_metrics(INTERIM_DATA_DIR / "thinning_diversity_metrics.csv","thinning")
    evolution_metrics(INTERIM_DATA_DIR / "thickening_diversity_metrics.csv","thickening")

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
