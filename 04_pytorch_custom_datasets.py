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
from typing import Tuple, Dict, List


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
BATCH_SIZE = 32

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

image, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n {image}")
print(f"Image label:\n {label}")
print(f"Image class name:\n {class_names[label]}")
print(f"Image datatype: {image.dtype}")
print(f"Image datatype: {image.shape}")
print(f"Label Data type: {type(label)}")

train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

target_dir = train_dir
# create function to get the class names using os.scandir() to traverse directory, and raise an error if class names aren't found. turn class names into a dict and a list and return them

class_names_found = sorted(
    [entry.name for entry in list(os.scandir(target_dir))])


def get_class_names(target_dir: str) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(entry.name for entry in os.scandir(
        target_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"couldn't find any classes in {target_dir}... please check file structure")
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return (class_to_idx, classes)
# loading image data with custom dataset


class PizzaSteakSushiDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: str, transform=None):
        super().__init__()
        # get all images in sub dir of path
        self.image_paths = list(Path(image_paths).glob("*/*.jpg"))
        self.transform = transform
        self.class_to_idx, self.classes = get_class_names(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.image_paths[index])

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        label = self.class_to_idx[self.image_paths[index].parent.name]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])


def display_random_image(dataset: torch.utils.data.Dataset, classes: List[str] = None,
                         n: int = 10, display_shape: bool = True, seed: int = None):
    if seed:
        random.seed(seed)
    if n > 10:
        n = 10
        display_shape = False
        print(f"Too large number of displays")
    random_image_indices = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, target_sample in enumerate(random_image_indices):
        image, label = dataset[target_sample][0], dataset[target_sample][1]
        # set color channel to be last instead of first
        image_adj = image.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(image_adj)
        plt.axis('off')
        if classes:
            title = f"Class: {classes[label]}"
            if display_shape:
                title = title + f"\nshape: {image_adj.shape}"
            plt.title(title)
    plt.show()


if __name__ == "__main__":
    # plot_transformed_images(image_paths=image_path_list,
    #                         transform=data_transform, n=3, seed=42)
    train_data_custom = PizzaSteakSushiDataset(
        image_paths=train_dir, transform=train_transforms)
    test_data_custom = PizzaSteakSushiDataset(
        image_paths=test_dir, transform=test_transforms)
    display_random_image(dataset=train_data_custom,
                         classes=train_data_custom.classes, n=5, seed=None)
    print("E")
