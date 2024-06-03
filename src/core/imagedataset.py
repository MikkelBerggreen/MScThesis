import torch
from torch.utils.data import Dataset
import pathlib
from pathlib import Path
from utils.config_loader import ConfigLoader

# Custom dataset class used to handle loading and transformations of images only when the samples is accessed from a dataloader.
# This is done to prevent storing multiple copies of images when creating the dataloaders.
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None):
        self.image_paths = image_paths
        self.labels = labels
        self.config = ConfigLoader()

    def __len__(self):
        return len(self.image_paths)

    # Necessary method for retrieving elements from the dataloader
    def __getitem__(self, idx):
        # Open image from path
        base_path = Path(__file__).resolve().parent
        image_path = (base_path / f"../../preprocessed_{self.image_paths[idx]}").resolve()
        image_path = pathlib.PureWindowsPath(image_path).as_posix().replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
        image_tensor = torch.load(image_path)

        label = self.labels[idx]

        return image_tensor, label