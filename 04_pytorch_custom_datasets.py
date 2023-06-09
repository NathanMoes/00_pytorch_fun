import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from helper_functions import plot_predictions, plot_decision_boundary, accuracy_fn
import torchinfo
from torchinfo import summary
from timeit import default_timer as timer
import torchvision


random.seed(42)
CPU_COUNT = 0


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


device = "cuda" if torch.cuda.is_available() else "cpu"

# setup path to data
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

#
if image_path.is_dir():
    print(f"{image_path} directory already exists... skipping download")
else:
    print(f"{image_path} does not exist, creating it now")
    image_path.mkdir(parents=True, exist_ok=True)

with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get(
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    )
    print(f"Downloading pizza steak sushi data...")
    f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza sushi steak")
    zip_ref.extractall(image_path)


train_dir = image_path / "train"
test_dir = image_path / "test"

image_path_list = list(image_path.glob("*/*/*.jpg"))
rnd_image_path = random.choice(image_path_list)
image_class = rnd_image_path.parent.stem
img = Image.open(rnd_image_path)
print(f"random image path: {rnd_image_path} | and class name: {image_class}")
print(img)

img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class {image_class} | Image shape: {img_as_array.shape}")
plt.axis(False)
plt.show()

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
IMAGE_OUT_MULT = ((IMAGE_HEIGHT / 2) / 2) * ((IMAGE_HEIGHT / 2) / 2)

data_transform = transforms.Compose([
    # resize to 64 x 64
    transforms.Resize(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
    # flip horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # turn image into torch tensor
    transforms.ToTensor()
])

data_transform(img)


def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)
            # transformed
            transformed_img = transform(f).permute(1, 2, 0)  # C,H,W to H,W,C
            ax[1].imshow(transformed_img)
            ax[1].set_title(f"Transformed\nSize: {transformed_img.shape}")
            ax[1].axis(False)
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()


train_data = datasets.ImageFolder(
    root=train_dir, transform=data_transform, target_transform=None)

test_data = datasets.ImageFolder(
    root=test_dir, transform=data_transform, target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

image, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n {image}")
print(f"Image label:\n {label}")
print(f"Image class name:\n {class_names[label]}")
print(f"Image datatype: {image.dtype}")
print(f"Image datatype: {image.shape}")
print(f"Label Data type: {type(label)}")

train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_COUNT)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=CPU_COUNT)

target_dir = train_dir
# create function to get the class names using os.scandir() to traverse directory, and raise an error if class names aren't found. turn class names into a dict and a list and return them

class_names_found = sorted(
    [entry.name for entry in list(os.scandir(target_dir))])


def get_class_names(target_dir: str) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(entry.name for entry in os.scandir(
        target_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"couldn't find any classes in {target_dir}... please check file structure")
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return (class_to_idx, classes)
# loading image data with custom dataset


class PizzaSteakSushiDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: str, transform=None):
        super().__init__()
        # get all images in sub dir of path
        self.image_paths = list(Path(image_paths).glob("*/*.jpg"))
        self.transform = transform
        self.class_to_idx, self.classes = get_class_names(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.image_paths[index])

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        label = self.class_to_idx[self.image_paths[index].parent.name]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])


train_data_custom = PizzaSteakSushiDataset(
    image_paths=train_dir, transform=train_transforms)
test_data_custom = PizzaSteakSushiDataset(
    image_paths=test_dir, transform=test_transforms)
train_dataloader_custom = DataLoader(
    dataset=train_data_custom, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_COUNT)
test_dataloader_custom = DataLoader(
    dataset=test_data_custom, batch_size=BATCH_SIZE, shuffle=False, num_workers=CPU_COUNT)


def display_random_image(dataset: torch.utils.data.Dataset, classes: List[str] = None,
                         n: int = 10, display_shape: bool = True, seed: int = None):
    if seed:
        random.seed(seed)
    if n > 10:
        n = 10
        display_shape = False
        print(f"Too large number of displays")
    random_image_indices = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, target_sample in enumerate(random_image_indices):
        image, label = dataset[target_sample][0], dataset[target_sample][1]
        # set color channel to be last instead of first
        image_adj = image.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(image_adj)
        plt.axis('off')
        if classes:
            title = f"Class: {classes[label]}"
            if display_shape:
                title = title + f"\nshape: {image_adj.shape}"
            plt.title(title)
    plt.show()


simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data_simple = datasets.ImageFolder(
    root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(
    root=test_dir, transform=simple_transform)

simple_train_dataloader = DataLoader(
    dataset=train_data_simple, batch_size=BATCH_SIZE, num_workers=CPU_COUNT, shuffle=True)
simple_test_dataloader = DataLoader(
    dataset=test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=CPU_COUNT
)


class TinyVGG(torch.nn.Module):
    def __init__(self, inputShape: int, outputShape: int, imageDim: int,  hiddenNodes: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inputShape,
                      out_channels=hiddenNodes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenNodes, out_channels=hiddenNodes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenNodes, out_channels=hiddenNodes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenNodes, out_channels=hiddenNodes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenNodes*int(IMAGE_OUT_MULT),
                      out_features=outputShape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_layer(self.conv2(self.conv1(x)))


def test_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: torch.nn.Module, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)  # forward pass
            loss = loss_fn(test_pred_logits, y)  # calc loss
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / \
                len(test_pred_labels)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, device=device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # send to device
        y_pred = model(X)  # forward pass
        loss = loss_fn(y_pred, y)  # calc loss
        train_loss += loss.item()
        # train_acc += accuracy_fn(y, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    # print(f"Train loss: {train_loss} | Train acc: {train_acc}%\n")
    return train_loss, train_acc


def train(epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader,
          loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, model: torch.nn.Module, device=device):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader,
                                           loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}\n Test loss {test_loss} | Test acc: {test_acc}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results


def plot_loss_curves(results: Dict[str, List[float]]):
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Train accuracy")
    plt.plot(epochs, test_accuracy, label="Test accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


train_trainsform_trivial = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_simple = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor()
])

train_data_augmented = datasets.ImageFolder(
    root=train_dir, transform=train_trainsform_trivial)
test_data_simple = datasets.ImageFolder(
    root=test_dir, transform=test_transform_simple)

train_dataloader_augmented = DataLoader(
    dataset=train_data_augmented, batch_size=BATCH_SIZE, shuffle=True, num_workers=CPU_COUNT)
test_dataloader_simple = DataLoader(
    dataset=test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=CPU_COUNT)

custom_image_path = data_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already downloaded")

custom_image_uint8 = torchvision.io.read_image(
    str(custom_image_path)).type(torch.float32)
custom_image = torchvision.io.read_image(
    str(custom_image_path)).type(torch.float32) / 255
custom_image_transform = transforms.Compose([
    transforms.Resize(
        size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
])
custom_image_transformed = custom_image_transform(custom_image)
# resize to 64 by 64 image and set as pytorch float 32 tensor


# custom_image = Image.open(custom_image_path)
# custom_image = custom_image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
# custom_image = np.array(custom_image)
# custom_image = torch.tensor(custom_image)
# custom_image = custom_image.permute(2, 0, 1)
# custom_image = custom_image.unsqueeze(0)
# custom_image = custom_image.to(device)


# def predict_custom_image():
#     return


if __name__ == "__main__":
    # plot_transformed_images(image_paths=image_path_list,
    #                         transform=train_transform, n=5, seed=42)
    # display_random_image(dataset=train_data_custom,
    #                      classes=train_data_custom.classes, n=5, seed=None)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # input = num color channels
    # output = num classes
    # torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    first_model = TinyVGG(inputShape=3, hiddenNodes=16,
                          outputShape=len(class_names), imageDim=224).to(device)
    summary(first_model, input_size=[1, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
    # epoch batch train loop for pytorch model
    first_model.eval()
    with torch.inference_mode():
        custom_image_pred = first_model(
            custom_image_transformed.unsqueeze(0).to(device))
        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
        class_num = torch.argmax(custom_image_pred_probs, dim=1)
        print(
            f"Custom image prediction: {class_names[class_num]} with {custom_image_pred_probs}% chance")
    EPOCHS = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(first_model.parameters(), lr=0.001)
    start_time = timer()
    model_results = train(model=first_model, loss_fn=loss_fn,
                          optimizer=optimizer, epochs=EPOCHS, train_dataloader=train_dataloader_augmented, test_dataloader=test_dataloader_simple)
    end_time = timer()
    print(f"Training took {end_time-start_time} seconds")
    plot_loss_curves(model_results)
    first_model.eval()
    with torch.inference_mode():
        custom_image_pred = first_model(
            custom_image_transformed.unsqueeze(0).to(device))
        custom_image_pred_probs = torch.softmax(
            custom_image_pred, dim=1).to("cpu")
        class_num = torch.argmax(custom_image_pred_probs, dim=1).to("cpu")
        print(
            f"Custom image prediction: {class_names[class_num]} with {custom_image_pred_probs}% chance [pizza, steak, sushi]")
