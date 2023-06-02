import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import requests
from torchmetrics import Accuracy


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

torchMetricAcc = Accuracy(task='multiclass', num_classes=4).to(device)

# make samples
n_samples = 1000
n_features = 5

# set init hyp param
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# init data
X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES,
                            centers=NUM_CLASSES, cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = X_blob_train.to(
    device), X_blob_test.to(device), y_blob_train.to(device), y_blob_test.to(device)

# visualise
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()


def acc_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


class multiClassificationModule(nn.Module):

    def __init__(self, input_features, output_features, hidden_units=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # create 2 nn layers for data inp and out
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


if __name__ == "__main__":
    print('e')
    epochs = 100
    # create model
    model_0 = multiClassificationModule(
        input_features=2, output_features=4, hidden_units=8).to(device=device)
    # select loss fn
    loss_fn = nn.CrossEntropyLoss()
    # select optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # train loop
    for epoch in range(epochs):
        model_0.train()
        y_train_logits = model_0(X_blob_train)
        y_pred = torch.softmax(y_train_logits, dim=1).argmax(
            dim=1)
        loss = loss_fn(y_train_logits, y_blob_train)
        acc = acc_fn(y_true=y_blob_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_0.eval()
        with torch.inference_mode():
            y_logits = model_0(X_blob_test)
            y_pred_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_pred_probs, dim=1)
            test_loss = loss_fn(y_logits, y_blob_test)
            test_acc = acc_fn(y_true=y_blob_test, y_pred=y_preds)
            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch} | loss {loss}, Acc: {acc} | Test loss {test_loss}, Test Acc: {test_acc}")
                # print(y_preds)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_blob_train, y_blob_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_blob_test, y_blob_test)
    plt.show()
    print(torchMetricAcc(y_preds, y_blob_test))
