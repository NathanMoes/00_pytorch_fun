import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image


random.seed(42)


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


device = "cuda" if torch.cuda.is_available() else "cpu"

# setup path to data
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#
if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")
else:
    print(f"{image_path} does not exist, creating it now")
    image_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get(
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    )
    print(f"Downloading pizza steak sushi data...")
    f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza sushi steak")
    zip_ref.extractall(image_path)


train_dir = image_path / "train"
test_dir = image_path / "test"

image_path_list = list(image_path.glob("*/*/*.jpg"))
rnd_image_path = random.choice(image_path_list)
image_class = rnd_image_path.parent.stem
img = Image.open(rnd_image_path)
print(f"random image path: {rnd_image_path} | and class name: {image_class}")
print(img)

if __name__ == "__main__":
    print("E")
