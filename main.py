import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path


device = "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
torch.device(device=device)
MODEL_PATH = Path("models")


class LinearRegModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float32))
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.SGD(
            params=self.parameters(), lr=0.001)
        start = 0
        end = 1
        step = 0.02
        X = torch.arange(start, end, step).unsqueeze(dim=1)
        y = self.weights * X + self.bias
        train_split = int(0.8 * len(X))
        self.X_train, self.y_train = X[:train_split], y[:train_split]
        self.X_test, self.y_test = X[train_split:], y[train_split:]
        self.epochs = 1700
        self.epoch_count = []
        self.loss_values = []
        self.test_loss_values = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

    def test_output(self, epoch, loss):
        with torch.inference_mode():  # trun off tracking of un needed things
            test_prediction = self.forward(self.X_test)
            test_loss = self.loss_fn(test_prediction, self.y_test)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | loss: {loss} | Test loss: {test_loss}")
            print(f"Weight: {self.weights} | bias: {self.bias}")
        return test_loss

    def print_loss_curve(self):
        with torch.inference_mode():
            plt.plot(self.epoch_count, self.loss_values,
                     label="Train loss")
            plt.plot(self.epoch_count, self.test_loss_values,
                     label="Test loss")
            plt.title('Training and test loss curves')
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()

    def train_loop(self):

        for epoch in range(self.epochs):
            self.epoch_count.append(epoch)
            self.train()  # turn on grads for needed updates
            y_pred = self.forward(self.X_train)  # forward pass
            loss = self.loss_fn(y_pred, self.y_train)  # calc loss
            self.loss_values.append(loss)
            self.optimizer.zero_grad()  # optimse base on gradient?
            loss.backward()  # back prop error
            self.optimizer.step()  # perform grad desc
            self.eval()  # enable testing
            test_loss = self.test_output(epoch=epoch, loss=loss)
            self.test_loss_values.append(test_loss)
            # if epoch % 1000 == 0:
            #     print(
            #         f"Loss: {loss}, weight:{self.weights} and bias: {self.bias}")


def pytorchWorkFlow():

    weight = 0.7
    bias = 0.3

    torch.manual_seed(42)
    my_model = LinearRegModel()
    my_model.train_loop()
    my_model.print_loss_curve()

    # def plot_Predictions(train_data=X_train, train_label=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    #     print('e')
    #     plt.figure(figsize=(10, 7))
    #     plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    #     plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    #     if predictions is not None:
    #         plt.scatter(test_data, predictions, c='r',
    #                     s=4, label="Predictions")

    #     plt.legend(prop={"size": 14})
    #     plt.show()
    # with torch.inference_mode():
    #     y_preds = my_model(X_test)
    # plot_Predictions(predictions=y_preds)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "01_pytorch.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(obj=my_model.state_dict(), f=MODEL_SAVE_PATH)


if __name__ == "__main__":
    # pytorchWorkFlow()
    MODEL_SAVE_PATH = MODEL_PATH / "01_pytorch.pth"
    test = torch.load(MODEL_SAVE_PATH)
    newModel = LinearRegModel().load_state_dict(test)
    print(newModel)
