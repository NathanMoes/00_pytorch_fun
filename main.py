import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


device = "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
torch.device(device=device)


def pytorchWorkFlow():

    class LinearRegModel(nn.Module):
        def __init__(self, X_Train, y_train, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.weights = nn.Parameter(torch.randn(
                1, requires_grad=True, dtype=torch.float32))
            self.bias = nn.Parameter(torch.randn(
                1, requires_grad=True, dtype=torch.float32))
            self.loss_fn = nn.L1Loss()
            self.optimizer = torch.optim.SGD(
                params=self.parameters(), lr=0.01)
            self.epochs = 1000
            self.X_train = X_train

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias

        def train_loop(self):
            for epoch in range(self.epochs):
                self.train()
                y_pred = self.forward(self.X_train)
                loss = self.loss_fn(y_pred, y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.eval()

    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    torch.manual_seed(42)
    my_model = LinearRegModel(X_train, y_train=y_train)
    my_model.train_loop()

    def plot_Predictions(train_data=X_train, train_label=y_train, test_data=X_test, test_labels=y_test, predictions=None):
        print('e')
        plt.figure(figsize=(10, 7))
        plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
        plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

        if predictions is not None:
            plt.scatter(test_data, predictions, c='r',
                        s=4, label="Predictions")

        plt.legend(prop={"size": 14})
        plt.show()
    with torch.inference_mode():
        y_preds = my_model(X_test)
    plot_Predictions(predictions=y_preds)


if __name__ == "__main__":
    pytorchWorkFlow()
