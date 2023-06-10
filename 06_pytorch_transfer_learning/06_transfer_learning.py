import matplotlib.pyplot as plt
import torch
import torchvision
import sys
from torch import nn
from torchvision import transforms
import os

try:
    from going_modular.going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    os.system("git clone https://github.com/mrdbourke/pytorch-deep-learning")
    os.system("mv pytorch-deep-learning/going_modular .")
    os.system("rm -rf pytorch-deep-learning")
    from going_modular.going_modular import data_setup, engine


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


print(f"e")
