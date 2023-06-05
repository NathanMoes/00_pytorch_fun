import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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

img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class {image_class} | Image shape: {img_as_array.shape}")
plt.axis(False)
plt.show()

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

data_transform = transforms.Compose([
    # resize to 64 x 64
    transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    # flip horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # turn image into torch tensor
    transforms.ToTensor()
])

data_transform(img)


def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)
            # transformed
            transformed_img = transform(f).permute(1, 2, 0)  # C,H,W to H,W,C
            ax[1].imshow(transformed_img)
            ax[1].set_title(f"Transformed\nSize: {transformed_img.shape}")
            ax[1].axis(False)
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()


train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None)

test_data = datasets.ImageFolder(
    root=test_dir, transform=data_transform, target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

if __name__ == "__main__":
    plot_transformed_images(image_paths=image_path_list,
                            transform=data_transform, n=3, seed=42)
    print("E")
