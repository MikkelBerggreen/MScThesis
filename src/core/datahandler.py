from utils.utils import one_hot_encode_classes
from utils.config_loader import ConfigLoader
from core.imagedataset import ImageDataset
from utils.constants import RAW_EEG_FILE, EEG_CLASSES_FILE, EEG_CLASSES_TESTDATA_FILE, IMAGE_FILE, RANDOM_SEED
from sklearn.model_selection import train_test_split
from utils.transformations import transform_eeg
import torch
import numpy as np
import gzip
import pickle as p
from pathlib import Path

# Class used to handle EEG and image datasets from:
# - zipped numpy format
# - pytorch format
# - pickle format
# to:
# - training/validation/test sets
# - dataloaders
class DataHandler:
    def __init__(self, batch_size=None, changing_model_configs=None):
        self.config = ConfigLoader()
        self.dataset_name = self.config.current_dataset
        self.dataset_config = getattr(self.config.datasets, self.dataset_name)
        if changing_model_configs is None:
            self.batch_size = batch_size or self.dataset_config.batch_size
        else:
            self.batch_size = changing_model_configs['batch_size']
        self.image_data, self.eeg_data = self.load_dataset(changing_model_configs)
            
    # Load dataset from file
    def load_dataset(self, changing_model_configs=None):
        base_path = Path(__file__).resolve().parent
        eeg_file_path = (base_path / f"../../data/datasets/{self.dataset_name}/{RAW_EEG_FILE}").resolve()
        image_file_path = (base_path / f"../../data/datasets/{self.dataset_name}/{IMAGE_FILE}").resolve()
        
        # Load image data
        with open(image_file_path, 'rb') as f:
            image_data = p.load(f)
            # Reduce all classes to 12 samples each
            image_data = image_data[:22248]

        # Load output data dependent on if we want it to be EEG responses or image classes
        if self.config.task == 'eeg':
            with gzip.GzipFile(eeg_file_path, "r") as f:
                # Reduce all classes to 12 samples each
                eeg_data = torch.from_numpy(np.load(file=f))[:22248, :, :].float()
                eeg_data = transform_eeg(eeg_data, changing_model_configs)
        elif self.config.task == 'class':
            one_hot_encode_classes(image_data)
            class_encoding_file_path = (base_path / f"../../data/datasets/{self.dataset_name}/{EEG_CLASSES_FILE}").resolve() 
            with gzip.GzipFile(class_encoding_file_path, "r") as f:
                eeg_data = torch.from_numpy(np.load(file=f))

        return image_data, eeg_data
    
    # Split data into training and test
    def split_train_test(self, input_data, output_data):
        return train_test_split(
            input_data,
            output_data,
            test_size=self.config.datasets.general.test_size,
            random_state=RANDOM_SEED
        )

    # Split data into training and validation
    def split_train_validation(self, input_data, output_data):
        return train_test_split(
            input_data,
            output_data,
            test_size=self.config.datasets.general.validation_size,
            random_state=RANDOM_SEED
        )
    
    # Setup dataloader of x and y values
    def setup_dataloader(self, image_paths, eeg, shuffle=True):
        if "TCNAE" in self.config.current_model:
            # Transformations of data happens in the custom dataset (during the loading of data in training)      
            dataset = torch.utils.data.TensorDataset(eeg, eeg)
        else:
            dataset = ImageDataset(image_paths, labels=eeg)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return dataloader

    # Return train and validation dataloaders from dataset
    def get_train_val_loader(self):
        # Split into train and validation sets
        x_train, _, y_train, _ = self.split_train_test(self.image_data, self.eeg_data)
        x_train, x_val, y_train, y_val = self.split_train_validation(x_train, y_train)
        
        # Setup dataloaders
        train_dataloader = self.setup_dataloader(x_train, y_train, shuffle=True)
        val_dataloader = self.setup_dataloader(x_val, y_val, shuffle=False)

        return train_dataloader, val_dataloader
    
    # Return test dataloader from dataset
    def get_test_loader(self):
        # Split into train and test sets
        _, x_test, _, y_test = self.split_train_test(self.image_data, self.eeg_data)

        '''
        base_path = Path(__file__).resolve().parent
        save_path = (base_path / f"../../data/datasets/{self.config.current_dataset}/{EEG_CLASSES_TESTDATA_FILE}").resolve()
        if save_path.exists():
            return
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(f"{save_path}", 'wb') as f:
            np.save(f, y_test)
            print('correctly saved testset class encoding')
        '''
            
        # Setup dataloader
        test_dataloader = self.setup_dataloader(x_test, y_test, shuffle=False)

        return test_dataloader

    # Return shapes of input and output data before they're transformed to tensors and dataloaders
    def get_data_shapes(self, dataloader):
        # Fetch a single batch from the dataloader
        inputs, labels = next(iter(dataloader))

        return inputs.shape, labels.shape