import torch
from torch import nn
import requests
import pathlib
from pathlib import Path
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


device = "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
torch.device(device=device)
MODEL_PATH = Path("models")


# get dataset
train_data = datasets.FashionMNIST(
    root="data",  # download to where
    train=True,  # do we want train
    download=True,  # download?
    transform=ToTensor(),  # transform into tensor
    target_transform=None  # dont transform the labels
)

test_data = datasets.FashionMNIST(
    root="data",  # download to where
    train=False,  # do we want train
    download=True,  # download?
    transform=ToTensor(),  # transform into tensor
    target_transform=None  # dont transform the labels
)

# set batch size
BATCH_SIZE = 32

# turn dataset into batches
train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)


class_names = train_data.classes
# visualize data
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(label=class_names[label])
plt.show()
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.show()

# prepare dataloader

if __name__ == "__main__":
    print('e')
