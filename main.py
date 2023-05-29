import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

Tensor = torch.rand(size=(3, 4))
Tensor += 10
print(Tensor)
