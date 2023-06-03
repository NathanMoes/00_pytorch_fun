# https://www.learnpytorch.io/03_pytorch_computer_vision/#74-training-and-testing-model_2-using-our-training-and-test-functions
from helper_functions import plot_predictions, plot_decision_boundary, accuracy_fn
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
import timeit
from timeit import default_timer as timer
from tqdm.auto import tqdm


MODEL_PATH = Path("models")

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)


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

train_features_batch, train_labels_batch = next(
    iter(train_dataloader))


class_names = train_data.classes
# visualize data
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(label=class_names[label])
# plt.show()
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
# plt.show()

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


class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier_layer(x)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time} seconds")
    return total_time


def eval_model(model: torch.nn.Module, data_loader: DataLoader,
               loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    loss, acc = 0, 0
    model.to(device=device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc
    }


def test_loop(model: nn.Module, train_loss):
    test_loss, test_acc = 0.0, 0.0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(
        f"\nTrain loss: {train_loss} | Test loss: {test_loss}, test acc {test_acc}"
    )


def train_loop(model: nn.Module, loss_fn, optimizer, epochs):
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}")
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            model.train()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 400 == 0:
                print(
                    f"\nLooked at {batch * len(X)}/{len(train_dataloader.dataset)} samples"
                )
        train_loss /= len(train_dataloader)
        test_loop(model=model, train_loss=train_loss)
    trainEndTime = timer()
    print_train_time(
        start=trainTimeStart, end=trainEndTime, device=str(next(model.parameters()).device))
    model_res = eval_model(
        model=model, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    print(model_res)


def train_step(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               acc_fn: accuracy_fn,
               device: torch.device = device,
               epoch: int = 0) -> int:
    print(f"\nEpoch: {epoch}")
    train_loss, train_acc = 0, 0  # init loss and acc
    model.train()  # set to train mode
    for batch, (X, y) in enumerate(data_loader):  # loop over training batches
        X, y = X.to(device), y.to(device)  # set to device
        y_pred = model(X)  # do forward pass
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # from logits -> labels
        train_acc += acc_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            print(
                f"\nLooked at {batch * len(X)}/{len(data_loader.dataset)} samples"
            )
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss} | Train acc: {train_acc}%")
    return train_loss


def test_step(model: torch.nn.Module, data_loader: DataLoader,
              loss_fn: torch.nn.Module, acc_fn: accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device=device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)  # foreward pass
            test_loss += loss_fn(test_pred, y)  # calc loss
            # calc acc
            test_acc += acc_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss} | Test acc: {test_acc}%\n")


if __name__ == "__main__":
    test = None
    try:
        MODEL_SAVE_PATH = MODEL_PATH / "03_ComputerVisionMaxAcc.pth"
        test = torch.load(MODEL_SAVE_PATH)
    except:
        print("ooooopsi woopsi")
    if (not test):
        # TinyVGG is the model of v2
        torch.manual_seed(42)
        # model_0 = FashionMNISTModelV0(
        #     input_shape=784,  # 28 * 28 image
        #     hidden_units=10,
        #     output_shape=len(class_names)  # one for each class
        # ).to(device=device)
        model_0 = FashionMNISTModelV2(
            input_shape=1,  # number of color channels
            hidden_units=64,
            output_shape=len(class_names)  # one for each class
        ).to(device=device)
        # setup loss, optim, eval metric
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
        trainTimeStart = timer()
        epochs = 100  # 72, 76-99, 91-99.7, 95-99.9, 98-100
        for epoch in tqdm(range(epochs)):
            train_loss = train_step(model=model_0, data_loader=train_dataloader,
                                    loss_fn=loss_fn, optimizer=optimizer,
                                    acc_fn=accuracy_fn, device=device, epoch=epoch)
            test_step(model=model_0, data_loader=test_dataloader,
                      loss_fn=loss_fn, acc_fn=accuracy_fn, device=device)

        trainEndTime = timer()
        print_train_time(
            start=trainTimeStart, end=trainEndTime, device=str(next(model_0.parameters()).device))
        model_res = eval_model(
            model=model_0, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
        print(model_res)

        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
        # train_loop(model=model_0, loss_fn=loss_fn,
        #            optimizer=optimizer, epochs=epochs)
    else:
        model_from_save = FashionMNISTModelV2(input_shape=1,  # number of color channels
                                              hidden_units=100,
                                              output_shape=len(class_names))
        model_from_save.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(model_from_save.state_dict())
