import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


device = "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
torch.device(device=device)
MODEL_PATH = Path("models")

# make samples
n_samples = 1000

# create circles
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame(
    {
        "X1": X[:, 0],
        "X2": X[:, 1],
        "label": y
    }
)
plt.scatter(
    x=X[:, 0],
    y=X[:, 1],
    c=y,
    cmap=plt.cm.RdYlBu
)
plt.show()

X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


class classificationModule(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # create 2 nn layers for data inp and out
        self.input_layer = nn.Linear(in_features=2, out_features=5)
        self.output_layer = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.input_layer(x))


if __name__ == "__main__":
    model_0 = classificationModule().to(device)
    # model_0 = nn.Sequential(
    #     nn.Linear(in_features=2, out_features=5),
    #     nn.Linear(in_features=5, out_features=1)
    # ).to(device=device)

    def acc_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    with torch.inference_mode():
        untrain_pred = model_0(X_test.to(device))
