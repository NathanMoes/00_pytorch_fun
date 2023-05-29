import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = "cpu"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda'

Tensor = torch.rand(size=(3, 4))
Tensor += 10
print(Tensor)
