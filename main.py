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
    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    print(X[:10], y[:10], len(X), len(y))


if __name__ == "__main__":
    pytorchWorkFlow()
