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

from helper_functions import plot_predictions, plot_decision_boundary, accuracy_fn


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

train_features_batch, train_labels_batch = next(iter(train_dataloader))


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


# make model

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    model_0 = FashionMNISTModelV0(
        input_shape=784,  # 28 * 28 image
        hidden_units=10,
        output_shape=len(class_names)  # one for each class
    ).to(device=device)
    # setup loss, optim, eval metric
    loss_fn = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
    print('e')
