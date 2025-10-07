from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.morphomnist import io
import matplotlib.pyplot as plt
import numpy as np
import gzip
import struct

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
    input_images = io.load_idx(imgs_paths)

    with open(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/labels.csv","w") as labels_csvfile:
        labels_csvfile.write("img_id,label")
        for i,img in enumerate(tqdm(input_images)):
            labels_csvfile.write(f"\n{i}.png,{labels[i]}")
            plt.imsave(f"{PROCESSED_DATA_DIR}/morphomnist/{split}/{i}.png",img,cmap="gray")
    
    

@app.command()
def main():
    logger.info("Processing train MorphoMNIST dataset...")
    process_morpho_mnist(split="train")
    logger.success("Processing train MorphoMNIST complete.")

    logger.info("Processing test MorphoMNIST dataset...")
    process_morpho_mnist(split="test")
    logger.success("Processing test MorphoMNIST complete.")


if __name__ == "__main__":
    app()
