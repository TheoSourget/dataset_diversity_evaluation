from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.morphomnist import io as morphoio
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip
import struct
import ast
import tensorflow as tf
import glob
from sklearn.model_selection import GroupShuffleSplit,StratifiedGroupKFold
import cv2

app = typer.Typer()

def process_morpho_mnist(split):

    #Define train and test data path
    if split == "train":
        labels_path = f"{RAW_DATA_DIR}/morphomnist/plain/train-labels-idx1-ubyte.gz"
        imgs_paths = f"{RAW_DATA_DIR}/morphomnist/plain/train-images-idx3-ubyte.gz"
    else:
        labels_path = f"{RAW_DATA_DIR}/morphomnist/plain/t10k-labels-idx1-ubyte.gz"
        imgs_paths = f"{RAW_DATA_DIR}/morphomnist/plain/t10k-images-idx3-ubyte.gz"
    
    #Create the folder to save the processed images and labels
    Path(f"{PROCESSED_DATA_DIR}/morphomnist/{split}").mkdir(parents=True, exist_ok=True)
    
    #Load plain dataset labels
    with gzip.open(labels_path,'rb') as f:
        magic, size = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))

    #Load the images
    input_images = morphoio.load_idx(imgs_paths)

    with open(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/labels.csv","w") as labels_csvfile:
        labels_csvfile.write("img_id,label")
        for i,img in enumerate(tqdm(input_images)):
            labels_csvfile.write(f"\n{i}.png,{labels[i]}")
            plt.imsave(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/{i}.png",img,cmap="gray")

def padchest_filter_and_process_labels():
    base_df = pd.read_csv(f'{RAW_DATA_DIR}/padchest/labels.csv',index_col=0)
        
    # Excluding NaNs in the labels
    df_no_nan = base_df[~base_df["Labels"].isna()]
    # Excluding labels including the 'suboptimal study' label
    df_no_clear_label = df_no_nan[~df_no_nan["Labels"].str.contains('suboptimal study')]
    df_no_clear_label = df_no_clear_label[~df_no_nan["Labels"].str.contains('exclude')]
    df_no_clear_label = df_no_clear_label[~df_no_nan["Labels"].str.contains('Unchanged')]

    # Keeping only the PA, AP and AP_horizontal projections
    df_view = df_no_clear_label[(df_no_clear_label['Projection'] == 'PA') | (df_no_clear_label['Projection'] == 'AP') | (df_no_clear_label['Projection'] == 'AP_horizontal')]

    # Stripping and lowercasing all individual labels
    stripped_lowercased_labels = []

    for label_list in list(df_view['Labels']):
        label_list = ast.literal_eval(label_list)
        prepped_labels = []
        
        for label in label_list:
            if label != '':
                new_label = label.strip(' ').lower()   # Stripping and lowercasing
                prepped_labels.append(new_label)
        
        # Removing label duplicates in this appending
        stripped_lowercased_labels.append(list(set(prepped_labels)))

    # Applying it to the preprocessed dataframe
    df_view['Labels'] = stripped_lowercased_labels    

    df_view['label'] = df_view['Labels'].apply(lambda label_list: 1 if "pneumothorax" in label_list else 0)
    df_to_save = df_view.reset_index(drop=True)
    df_to_save.to_csv(f"{PROCESSED_DATA_DIR}/padchest/processed_labels.csv",sep=",")



def padchest_process_images():
    #Load labels
    labels = pd.read_csv(f'{PROCESSED_DATA_DIR}/padchest/processed_labels.csv')
    
    #Get images present at input_filepath
    images_path = glob.glob(f"{RAW_DATA_DIR}/padchest/**/*.png",recursive=True)
    image_names = [path.split('/')[-1] for path in images_path]
    for idx,i_name in enumerate(tqdm(image_names)):
        #Resize the image and save it in the processed folder
        if i_name in labels["ImageID"].unique():
            try:
                img = np.expand_dims(io.imread(images_path[idx]),-1)
                max_value = np.max(img)
                if max_value==0:
                    print(f"Discarding {i_name}, black images")
                    continue 
                img = tf.image.resize_with_pad(img, 512, 512)
                img = img/max_value
                tf.keras.utils.save_img(f"{PROCESSED_DATA_DIR}/padchest/images/{i_name}", img, scale=True, data_format="channels_last")  
            except Exception as e:
                print(f"Discarding {i_name}, error {e}")

def train_test_split_padchest():
    #Load labels
    labels = pd.read_csv(f'{PROCESSED_DATA_DIR}/padchest/processed_labels.csv')

    #Split the data into 80/20 groups, ensuring a single patient is in a single split to avoid data leakage
    splitter = StratifiedGroupKFold(n_splits=5)
    train_test_split = splitter.split(X=labels["ImageID"],y=labels["label"], groups=labels['PatientID'])
    train_idx, test_idx = next(train_test_split)
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    test_labels = labels.iloc[test_idx].reset_index(drop=True)
    
    train_labels.to_csv(f"{PROCESSED_DATA_DIR}/padchest/train_labels.csv",sep=",")
    test_labels.to_csv(f"{PROCESSED_DATA_DIR}/padchest/test_labels.csv",sep=",")


def process_padchest():
    Path(f"{PROCESSED_DATA_DIR}/padchest/images").mkdir(parents=True, exist_ok=True)
    logger.info("Processing labels for PadChest dataset...")
    padchest_filter_and_process_labels()
    logger.info("Processing images for PadChest dataset...")
    padchest_process_images()
    logger.info("Train-Test split of PadChest...")
    train_test_split_padchest()
    
@app.command()
def main():
    logger.info("Processing train MorphoMNIST dataset...")
    process_morpho_mnist(split="train")
    logger.success("Processing train MorphoMNIST complete.")

    logger.info("Processing test MorphoMNIST dataset...")
    process_morpho_mnist(split="test")
    logger.success("Processing test MorphoMNIST complete.")

    logger.info("Processing PadChest dataset...")
    process_padchest()
    logger.success("Processing PadChest complete.")

if __name__ == "__main__":
    app()
