import random
from PIL import Image
from timeit import default_timer as timer
import torchinfo
from torchvision import transforms
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch import nn
import torchvision
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import zipfile
import requests
import torch
device = "cpu"

BATCH_SIZE = 32

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
torch.device(device=device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


set_seeds()


def download_data(source: str = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                  destination: str = "pizza_steak_sushi",
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path


image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


"""
Contains functions for training and testing a PyTorch model.
"""


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() /
                         len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter = None
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
      For example if training for epochs=2: 
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        # experiment tracking
        if writer is not None:
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={
                "train_loss": train_loss,
                "test_loss": test_loss}, global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={
                "train_acc": train_acc, "train_acc": test_acc}, global_step=epoch)
            writer.add_graph(model=model,
                             input_to_model=torch.randn(BATCH_SIZE, 3, 224, 224).to(device))
            writer.close()

    # Return the filled results at the end of the epochs
    return results


"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""


NUM_WORKERS = 0


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    test_transform: transforms.Compose = None,
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: torchvision transforms to perform on training and testing data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for number of workers per DataLoader.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, test_dataloader, class_names = \
          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               batch_size=32,
                               num_workers=4)
    """
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    if test_transform is not None:
        test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

# create a function that creates a SummaryWriter() instance


def create_writer(experiment_name: str, model_name: str, extra: str = None):
    """
    creates a summary writer instance to specific dir
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        # Create log directory path
        log_dir = os.path.join(
            "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def pred_and_plot_image(model: torch.nn.Module, class_names: List[str],
                        image_path: str, transform: torchvision.transforms = None,
                        image_size: Tuple[int, int] = (224, 224), device: torch.device = device):
    model.eval()
    model.to(device=device)
    ogImage = image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    else:
        image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])(image)
    image = image.unsqueeze(0)
    image = image.to(device=device)
    with torch.inference_mode():
        output = model(image)
        predicted_probs = torch.softmax(output, dim=1)
        predicted = torch.argmax(predicted_probs, dim=1)
        plt.figure()
        plt.imshow(ogImage)
        plt.title(
            f"Predicted class: {class_names[predicted]} | probability: {predicted_probs.max():.4f}")
        plt.axis("off")
        plt.show()


def visualize():
    # Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
    image_permuted = image.permute(1, 2, 0)

    # Index to plot the top row of patched pixels
    patch_size = 16
    plt.figure(figsize=(patch_size, patch_size))
    plt.imshow(image_permuted[:patch_size, :, :])
    # Setup hyperparameters and make sure img_size and patch_size are compatible
    img_size = 224
    patch_size = 16
    num_patches = img_size/patch_size
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    print(
        f"Number of patches per row: {num_patches}\nPatch size: {patch_size} pixels x {patch_size} pixels")

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=1,
                            ncols=img_size // patch_size,  # one column for each patch
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)

    # Iterate through number of patches in the top row
    for i, patch in enumerate(range(0, img_size, patch_size)):
        # keep height index constant, alter the width index
        axs[i].imshow(image_permuted[:patch_size, patch:patch+patch_size, :])
        axs[i].set_xlabel(i+1)  # set the label
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()
    # Setup hyperparameters and make sure img_size and patch_size are compatible
    img_size = 224
    patch_size = 16
    num_patches = img_size/patch_size
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    print(f"Number of patches per row: {num_patches}\
            \nNumber of patches per column: {num_patches}\
            \nTotal patches: {num_patches*num_patches}\
            \nPatch size: {patch_size} pixels x {patch_size} pixels")

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=img_size // patch_size,  # need int not float
                            ncols=img_size // patch_size,
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)

    # Loop through height and width of image
    # iterate through height
    for i, patch_height in enumerate(range(0, img_size, patch_size)):
        # iterate through width
        for j, patch_width in enumerate(range(0, img_size, patch_size)):

            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size,  # iterate through height
                                            patch_width:patch_width+patch_size,  # iterate through width
                                            :])  # get all color channels

            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1,
                                 rotation="horizontal",
                                 horizontalalignment="right",
                                 verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

    # Set a super title
    fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    plt.show()


# Download pizza, steak, sushi images from GitHub
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

IMAGE_SIZE = 224
# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"
manual_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
# load in train, test dataloaders and class_names from create dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=manual_transform, batch_size=BATCH_SIZE, num_workers=0)
image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Class: {class_names[label]}")
plt.axis(False)
plt.show()


height = 224
width = 224
color_channels = 3
patch_size = 16
num_patches = int((height * width) / patch_size**2)
embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (num_patches, patch_size**2 * color_channels)

# equation 1L split data into patches
# equation 2L apply convolutional layer to each patch
# equation 3L apply max pooling to each patch
# equation 4L concatenate all patches together
# equation 5L apply fully connected layer to concatenated patches
# equation 6L make prediction


if __name__ == "__main__":
    visualize()
    pass
