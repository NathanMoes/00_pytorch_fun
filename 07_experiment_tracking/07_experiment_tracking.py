"""
Contains various utility functions for PyTorch model training and saving.
"""
import requests
import zipfile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import torch
from pathlib import Path
import torchvision
from torch import nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import sys
from torchvision import transforms
import torchinfo
from timeit import default_timer as timer
from PIL import Image
import random

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


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# Get num out features (one for each class pizza, steak, sushi)


# Create an EffNetB0 feature extractor


def create_effnetb0(out_feats):
    # 1. Get the base mdoel with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_feats)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an EffNetB2 feature extractor


def create_effnetb2(out_feats):
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    set_seeds()

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=out_feats)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb2"
    print(f"[INFO] Created new {model.name} model.")
    return model


data_10_percent_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
data_20_percent_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip", destination="pizza_steak_sushi_20_percent")
image_path = data_10_percent_path
train_dir_20 = data_20_percent_path / "train"
test_dir = image_path / "test"
train_dir = image_path / "train"

manual_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize,
])


BATCH_SIZE = 32
EPOCHS = 5

# 10% data
train_dataloader_10_percent, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=manual_transform, num_workers=0, batch_size=BATCH_SIZE)
# 20% dataloaders
train_dataloader_20_percent, test_dataloader, class_names = create_dataloaders(
    train_dir=train_dir_20, test_dir=test_dir, transform=manual_transform, num_workers=0, batch_size=BATCH_SIZE)


def run_experiments():
    # 1. Create epochs list
    num_epochs = [5, 10]

    # 2. Create models list (need to create a new model for each experiment)
    models = ["effnetb0", "effnetb2"]

    # 3. Create dataloaders dictionary for various dataloaders
    train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                         "data_20_percent": train_dataloader_20_percent}

    # 1. Set the random seeds
    set_seeds(seed=42)

    # 2. Keep track of experiment numbers
    experiment_number = 0

    # 3. Loop through each DataLoader
    for dataloader_name, train_dataloader in train_dataloaders.items():

        # 4. Loop through each number of epochs
        for epochs in num_epochs:

            # 5. Loop through each model name and create a new model based on the name
            for model_name in models:

                # 6. Create information print outs
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                # 7. Select the model
                if model_name == "effnetb0":
                    # creates a new model each time (important because we want each experiment to start from scratch)
                    model = create_effnetb0(len(class_names))
                else:
                    # creates a new model each time (important because we want each experiment to start from scratch)
                    model = create_effnetb2(len(class_names))

                # 8. Create a new loss and optimizer for every model
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(
                    params=model.parameters(), lr=0.001)

                # 9. Train target model with target dataloaders and track experiments
                train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=epochs,
                      device=device,
                      writer=create_writer(experiment_name=dataloader_name,
                                           model_name=model_name,
                                           extra=f"{epochs}_epochs"))

                # 10. Save the model to file so we can get back the best model
                save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                save_model(model=model,
                           target_dir="models",
                           model_name=save_filepath)
                print("-"*50 + "\n")


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


if __name__ == "__main__":
    # Setup the best model filepath
    best_model_path = "models/07_effnetb2_data_20_percent_10_epochs.pth"

    # Instantiate a new instance of EffNetB2 (to load the saved state_dict() to)
    best_model = create_effnetb2(len(class_names))

    # Load the saved best model state_dict()
    best_model.load_state_dict(torch.load(best_model_path))
    effnetb2_model_size = Path(best_model_path).stat().st_size // (1024*1024)
    print(
        f"EfficientNetB2 feature extractor model size: {effnetb2_model_size} MB")
    import random
    num_images_to_plot = 10
    # get all test image paths from 20% dataset
    test_image_path_list = list(
        Path(data_20_percent_path / "test").glob("*/*.jpg"))
    test_image_path_sample = random.sample(population=test_image_path_list,
                                           k=num_images_to_plot)  # randomly select k number of images

    # Iterate through random test image paths, make predictions on them and plot them
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=best_model,
                            image_path=image_path,
                            class_names=class_names,
                            image_size=(224, 224))

    # Download custom image

    # Setup custom image path
    custom_image_path = Path("data/04-pizza-dad.jpeg")

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
    pred_and_plot_image(model=best_model,
                        image_path=custom_image_path,
                        class_names=class_names)
    pass
