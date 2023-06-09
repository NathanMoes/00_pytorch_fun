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


def plot_loss_curves(results: Dict[str, List[float]]):
    """
    Plots the loss and accuracy curves for the training and test set
    """
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


# Create the Conv2d layer with hyperparameters from the ViT paper
conv2d = nn.Conv2d(in_channels=3,  # number of color channels
                   out_channels=768,  # from Table 1: Hidden size D, this is the embedding size
                   # could also use (patch_size, patch_size)
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)

# Create flatten layer
# 1,768,14,14
flatten = nn.Flatten(start_dim=2,  # flatten feature_map_height (dimension 2) 14
                     end_dim=3)  # flatten feature_map_width (dimension 3) 14


# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
        return x_flattened.permute(0, 2, 1)


set_seeds()

# Create an instance of patch embedding layer
patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)

# Pass a single image through
print(f"Input image shape: {image.unsqueeze(0).shape}")
# add an extra batch dimension on the 0th index, otherwise will error
patch_embedded_image = patchify(image.unsqueeze(0))
print(f"Output patch embedding shape: {patch_embedded_image.shape}")


# create class token embedding
# Get the batch size and embedding dimension
batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

# Create the class token embedding as a learnable parameter that shares the same size as the embedding dimension (D)
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),  # [batch_size, number_of_tokens, embedding_dimension]
                           requires_grad=True)  # make sure the embedding is learnable

# Show the first 10 examples of the class_token
print(class_token[:, :, :10])

# Print the class_token shape
print(
    f"Class token shape: {class_token.shape} -> [batch_size, number_of_tokens, embedding_dimension]")

# Add the class token embedding to the front of the patch embedding
patch_embedded_image_with_class_embedding = torch.cat((class_token, patch_embedded_image),
                                                      dim=1)  # concat on first dimension

# Print the sequence of patch embeddings with the prepended class token embedding
print(patch_embedded_image_with_class_embedding)
print(
    f"Sequence of patch embeddings with class token prepended shape: {patch_embedded_image_with_class_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

# Calculate N (number of patches)
number_of_patches = int((height * width) / patch_size**2)

# Get embedding dimension
embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

# Create the learnable 1D position embedding
position_embedding = nn.Parameter(torch.ones(1,
                                             number_of_patches+1,
                                             embedding_dimension),
                                  requires_grad=True)  # make sure it's learnable

# Show the first 10 sequences and 10 position embedding values and check the shape of the position embedding
print(position_embedding[:, :10, :10])
print(
    f"Position embeddding shape: {position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

# Add the position embedding to the patch and class token embedding
patch_and_position_embedding = patch_embedded_image_with_class_embedding + \
    position_embedding
print(patch_and_position_embedding)
print(
    f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

set_seeds()

# 1. Set patch size
patch_size = 16

# 2. Print shape of original image tensor and get the image dimensions
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

# 3. Get image tensor and add batch dimension
x = image.unsqueeze(0)
print(f"Input image with batch dimension shape: {x.shape}")

# 4. Create patch embedding layer
patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size=patch_size,
                                       embedding_dim=768)

# 5. Pass image through patch embedding layer
patch_embedding = patch_embedding_layer(x)
print(f"Patching embedding shape: {patch_embedding.shape}")

# 6. Create class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                           requires_grad=True)  # make sure it's learnable
print(f"Class token embedding shape: {class_token.shape}")

# 7. Prepend class token embedding to patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(
    f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

# 8. Create position embedding
number_of_patches = int((height * width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                  requires_grad=True)  # make sure it's learnable

# 9. Add position embedding to patch embedding with class token
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(
    f"Patch and position embedding shape: {patch_and_position_embedding.shape}")

# 1. Create a class that inherits from nn.Module


class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1

    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)  # does our batch dimension come first?

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # query embeddings
                                             key=x,  # key embeddings
                                             value=x,  # value embeddings
                                             need_weights=False)  # do we need the weights or just the layer outputs?
        return attn_output


# Create an instance of MSABlock
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768,  # from Table 1
                                                             num_heads=12)  # from Table 1

# Pass patch and position image embedding through MSABlock
patched_image_through_msa_block = multihead_self_attention_block(
    patch_and_position_embedding)
print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")
print(f"Output shape MSA block: {patched_image_through_msa_block.shape}")


# 1. Create a class that inherits from nn.Module
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3

    def __init__(self,
                 embedding_dim: int = 768,  # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 dropout: float = 0.1):  # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            # "Dropout, when used, is applied after every dense layer.."
            nn.Dropout(p=dropout)
        )

    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# Create an instance of MLPBlock
mlp_block = MLPBlock(embedding_dim=768,  # from Table 1
                     mlp_size=3072,  # from Table 1
                     dropout=0.1)  # from Table 3

# Pass output of MSABlock through MLPBlock
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
print(f"Input shape of MLP block: {patched_image_through_mlp_block.shape}")
print(f"Output shape MLP block: {patched_image_through_mlp_block.shape}")


# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3

    def __init__(self,
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 # Amount of dropout for dense layers from Table 3 for ViT-Base
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0):  # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # 4. Create MLP block (equation 3)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    # 5. Create a forward() method
    def forward(self, x):

        # 6. Create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x) + x

        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x


# Create an instance of TransformerEncoderBlock
transformer_encoder_block = TransformerEncoderBlock()

# # Print an input and output summary of our Transformer Encoder (uncomment for full output)
# summary(model=transformer_encoder_block,
#         input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])
# 1. Create a ViT class that inherits from nn.Module


class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3

    def __init__(self,
                 img_size: int = 224,  # Training resolution from Table 3 in ViT paper
                 in_channels: int = 3,  # Number of channels in input image
                 patch_size: int = 16,  # Patch size
                 num_transformer_layers: int = 12,  # Layers from Table 1 for ViT-Base
                 embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
                 mlp_size: int = 3072,  # MLP size from Table 1 for ViT-Base
                 num_heads: int = 12,  # Heads from Table 1 for ViT-Base
                 attn_dropout: float = 0,  # Dropout for attention projection
                 mlp_dropout: float = 0.1,  # Dropout for dense/MLP layers
                 embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
                 num_classes: int = 1000):  # Default for ImageNet but can customize this
        super().__init__()  # don't forget the super().__init__()!

        # 3. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 11. Create a forward() method
    def forward(self, x):

        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        # "-1" means to infer the dimension (try this line on its own)
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        # run on each sample in a batch at 0 index
        x = self.classifier(x[:, 0])

        return x


# Create the same as above with torch.nn.TransformerEncoderLayer()
torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,  # Hidden size D from Table 1 for ViT-Base
                                                             nhead=12,  # Heads from Table 1 for ViT-Base
                                                             dim_feedforward=3072,  # MLP size from Table 1 for ViT-Base
                                                             dropout=0.1,  # Amount of dropout for dense layers from Table 3 for ViT-Base
                                                             activation="gelu",  # GELU non-linear activation
                                                             batch_first=True,  # Do our batches come first?
                                                             norm_first=True)  # Normalize first or after MSA/MLP layers?


# equation 1L split data into patches
# equation 2L apply convocational layer to each patch
# equation 3L apply max pooling to each patch
# equation 4L concatenate all patches together
# equation 5L apply fully connected layer to concatenated patches
# equation 6L make prediction


if __name__ == "__main__":
    # visualize()
    vit = ViT(num_classes=len(class_names)).to(device=device)
    from torchinfo import summary

    # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
    summary(model=vit,
            # (batch_size, color_channels, height, width)
            input_size=(32, 3, 224, 224),
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )
    set_seeds()

    # # Create a random tensor with same shape as a single image
    # # (batch_size, color_channels, height, width)
    # random_image_tensor = torch.randn(1, 3, 224, 224)

    # # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)

    # # Pass the random image tensor to our ViT instance
    # vit(random_image_tensor)
    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                 lr=3e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
                                 # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                 betas=(0.9, 0.999),
                                 weight_decay=0.3)  # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set the seeds
    set_seeds()

    # Train the model and save the training results to a dictionary
    results = train(model=vit,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=10,
                    device=device)
    plot_loss_curves(results)
    # requires torchvision >= 0.13, "DEFAULT" means best available
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(
        weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
    set_seeds()
    pretrained_vit.heads = nn.Linear(
        in_features=768, out_features=len(class_names)).to(device)
    # pretrained_vit # uncomment for model output
    # Download pizza, steak, sushi images from GitHub
    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                               destination="pizza_steak_sushi")
    # Setup train and test directory paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    # Setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                              test_dir=test_dir,
                                                                                              transform=pretrained_vit_transforms,
                                                                                              batch_size=32)  # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                                 lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the classifier head of the pretrained ViT feature extractor model
    set_seeds()
    pretrained_vit_results = train(model=pretrained_vit,
                                   train_dataloader=train_dataloader_pretrained,
                                   test_dataloader=test_dataloader_pretrained,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epochs=6,
                                   device=device)
    plot_loss_curves(pretrained_vit_results)
    save_model(model=pretrained_vit,
               target_dir="models",
               model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth")
    custom_image_path = image_path / "04-pizza-dad.jpeg"

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get(
                "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")

    # Predict on custom image
    pred_and_plot_image(model=pretrained_vit,
                        image_path=custom_image_path,
                        class_names=class_names)
    # fin
    pass
