# Dataset diversity evaluation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study on the diversity of dataset and its impact on performance. Using image, text and metadata metrics.
This repo is still a work in progress.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data                
    │   ├── __init__.py 
    │   ├── dataset.py               <- Code to preprocess the datasets
    │   ├── morphomnist_pytorch.py   <- Code of the pytorch dataset for morphomnist
    │   ├── padchest_pytorch.py      <- Code of the pytorch dataset for padchest          
    │   └── utils.py                 <- Code to downsample and bootstrap the data
    │
    ├── metrics                
    │   └── compute_metrics.py  <- Code to compute all the diversity metrics                    
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── train_morphomnist.py         <- Code to train on the morphomnist dataset 
    │   ├── predict_morphomnist.py       <- Code to run model inference with trained models on morphomnist
    │   ├── padchest_fairness.py         <- Code to compute AUC per subgroups on the padchest dataset        
    │   ├── predict_padchest.py          <- Code to run model inference with trained models on padchest        
    │   └── train_padchest.py            <- Code to train on the padchest dataset
    │
    ├── morphomnist <- MorphoMNIST source code (see https://github.com/dccastro/Morpho-MNIST/tree/main/morphomnist)                
    │   └── ...  
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Setup
Get the repo and create the environment
```sh
git clone https://github.com/TheoSourget/dataset_diversity_evaluation.git
python src/metrics/compute_metrics.py cxr
make create_environment
source activate dataset_diversity_evaluation
#Install the appropriate version of torch depending on your cuda version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#Install the rest of the dependencies
make requirements
```

## Data

This repository uses publicly available datasets that you need to download in order to run the experiments.

### MorphoMNIST
We use the MorphoMNIST data as a toy dataset used some of the code from [their repo](https://github.com/dccastro/Morpho-MNIST) to generate data. Please download the plain dataset from [this link provided by the authors](https://drive.google.com/uc?export=download&id=1-E3sbKtzN8NGNefUdky2NVniW1fAa5ZG). Place the plain folder into a /data/raw/morphomnist folder that you have to create.

### PadChest
Get the PadChest dataset from: https://bimcv.cipf.es/bimcv-projects/padchest/
Unzip every folder in the data/raw/padchest folder (let the images in the subfolder)
And put the labels and metadata csv in data/raw/padchest/labels.csv

### Preprocessing
To preprocess the data and generate the splits run the command:
```sh
make data
```
The processed data will be stored in /data/processed

## Metrics and trainings
To compute the metrics use the compute_metrics.py script with the parameter "cxr" or "morpho" to choose the dataset. For example:
```sh
python src/metrics/compute_metrics.py cxr
```

To train the models on PadChest, use the [train_padchest.py](https://github.com/TheoSourget/dataset_diversity_evaluation/blob/main/src/modeling/train_padchest.py) script. You can specify the dataset_config, the number of epochs, the batch size and learning rate, and the patience. For example:
```sh
python src/modeling/train_padchest.py --epochs 100 --lr 0.0001 --dataset-config ImagingDynamics
```

For MorphoMNIST, [train_padchest.py](https://github.com/TheoSourget/dataset_diversity_evaluation/blob/main/src/modeling/train_morphomnist.py) already launch the trainings for all configuration. You can specify the number of epochs, the batch size and learning rate, and the patience. For example:
```sh
python src/modeling/train_morphomnist.py --epochs 100
```

You can then use [predict_morphomnist.py](https://github.com/TheoSourget/dataset_diversity_evaluation/blob/main/src/modeling/predict_morphomnist.py) and [predict_padchest.py](https://github.com/TheoSourget/dataset_diversity_evaluation/blob/main/src/modeling/predict_padchest.py) to get predictions from the models. For example:
```sh
python src/modeling/predict_padchest.py
```
## Plots
To reproduce the figures run the command (work in progress):
```sh
make visualisations
```