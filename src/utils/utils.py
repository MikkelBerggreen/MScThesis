import os
import yaml
import time
import matplotlib.pyplot as plt
import torch
import wandb
import csv
import re
import numpy as np
import gzip
from pathlib import Path
from utils.constants import EEG_CLASSES_FILE
from utils.config_loader import ConfigLoader

# Load configs from configs.yaml (replaced by ConfigLoader)
def load_config():
    # Get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, 'configs.yaml')
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

# Get string representation of current date and time
def get_current_time():
    # Get the current time (local)
    t = time.localtime()
    current_time = time.strftime('%d_%m_%Y__%H_%M', t)

    # Make the day format have a _ instead of /
    current_time = current_time.replace("/","_")

    # Convert timestamp to string
    timeString = str(current_time)

    return timeString

# Plot loss curve of model training and validation losses
def plot_loss(losses, val_losses):
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.title(f'Training vs Validation Loss')
    plt.plot(losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Loss")
    plt.ylabel("Epoch")
    plt.legend()

    plt.show()

def save_model(current_dataset, current_model, max_epochs, timestamp, best_model_state=None, use_wandb=False):
    model_directory = f'../trained_models/'
    best_model_path = os.path.join(model_directory, f'{current_model}_{max_epochs}epochs_{timestamp}.pth')

    # Ensure the directory exists
    os.makedirs(model_directory, exist_ok=True)

    # Save model checkpoints using wandb when training is done (or stopped early)
    torch.save(best_model_state, best_model_path)

    if use_wandb:
        # Log the best model as a wandb artifact
        artifact = wandb.Artifact('model-checkpoint', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)

# Function used to create a one-hot encoding of the classes seen in the THINGS dataset, saving it in a file.
def one_hot_encode_classes(input_data):
    config = ConfigLoader()
    base_path = Path(__file__).resolve().parent
    save_path = (base_path / f"../../data/datasets/{config.current_dataset}/{EEG_CLASSES_FILE}").resolve() 
    tsv_file_path = (base_path / f"../../data/meta/things_concepts.tsv").resolve() 

    if save_path.exists():
        return
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tsv_file_path) as file:
        tsv_file = csv.DictReader(file, delimiter="\t")
        classes = []
        for row in tsv_file:
            classes.append(row['uniqueID'])

    # Setup output class data
    output_data = np.zeros((len(input_data), len(classes)), int)
    for idx, input in enumerate(input_data):
        reg = re.search(r"^stimuli\\([a-zA-Z0-9_-]*)\\.*$", input) # Regex test: https://regex101.com/r/Lo55pE/1
        image_class = reg.group(1)
        class_index = classes.index(image_class)
        output_data[idx, class_index] = 1

    with gzip.open(f"{save_path}", 'wb') as f:
        np.save(f, output_data)

# Helper function for getting the number of used channels from the list of channels in configs
def get_number_of_channels():
    config = ConfigLoader()
    dataset_name = config.current_dataset
    dataset_config = getattr(config.datasets, dataset_name)

    used_channels = config.used_channels
    if used_channels == 'All':
        number_of_channels = dataset_config.channels
    else:
        number_of_channels = len(used_channels)

    return number_of_channels
    
