from utils.utils import get_current_time, plot_loss, save_model
from utils.early_stopper import EarlyStopper
from utils.config_loader import ConfigLoader
from utils.model_setup import get_model_from_name, get_loss_function_from_name, get_optimizer_from_name
from core.datahandler import DataHandler
from core.tester import Tester

import torch
import wandb
import tqdm
from tqdm import tqdm
from torchsummary import summary

# Class used to:
# - setup dataloaders for a model
# - setup a model for training
# - train a model
class Trainer:
    def __init__(self, batch_size=None, lr=None, weight_decay=None, optimizer_name=None, dropout=None, max_epochs=None, avgpool_size=None):
        self.config = ConfigLoader()
        self.dataset_name = self.config.current_dataset
        self.dataset_config = getattr(self.config.datasets, self.dataset_name)

        # Set from wandb sweep config if provided, but with dot notation
        self.batch_size = batch_size or self.dataset_config.batch_size
        self.lr = lr or self.config.models.general.learning_rate
        self.weight_decay = weight_decay or self.config.models.general.weight_decay
        self.optimizer_name = optimizer_name or self.config.models.general.optimizer
        self.dropout = dropout or self.config.models.general.dropout
        self.max_epochs = max_epochs or self.config.models.general.max_epochs
        self.avgpool_size = avgpool_size or self.config.models.tcnae.avgpool_size

        self.losses = []
        self.val_losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup dataloaders of the current dataset
    def setup_data(self):
        data_handler = DataHandler(batch_size=self.batch_size)

        self.train_dataloader, self.val_dataloader = data_handler.get_train_val_loader()
        self.train_input_shape, self.train_output_shape = data_handler.get_data_shapes(self.train_dataloader)

        self.test_dataloader = data_handler.get_test_loader()
        self.test_input_shape, self.test_output_shape = data_handler.get_data_shapes(self.test_dataloader)

    def create_model(self, input_shape, output_shape, model_name, loss_function_name, lr, weight_decay, optimizer_name, dropout, avgpool_size):
        # Setup which model to use
        model = get_model_from_name(
            model_name, 
            input_shape, 
            output_shape, 
            dropout, 
            avgpool_size, 
            self.config.models.resnet.n_blocks
        )

        # Setup loss function
        criterion = get_loss_function_from_name(loss_function_name, self.config.task)
        
        # Setup optimizer
        optimizer = get_optimizer_from_name(optimizer_name, model, lr, weight_decay)
        
        return model, criterion, optimizer

    # Setup a model for training
    def setup_model(self):
        # Get dataloaders and data shapes
        self.setup_data()

        model, criterion, optimizer = self.create_model(
            self.train_input_shape, 
            self.train_output_shape, 
            self.config.current_model, 
            self.config.current_loss_function, 
            self.lr, self.weight_decay, 
            self.optimizer_name, 
            self.dropout,
            self.avgpool_size
        )
        model = model.to(self.device)

        # Freeze model layers in case of using a pretrained model
        if self.config.models.general.freeze and any('fc_layers' in name for name, _ in model.named_parameters()):
            for name, param in model.named_parameters():
                if "fc_layers" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Visual prints of model settings
        print("")
        print(f'Model setup done:')
        print(f'    Model: {self.config.current_model}') 
        print(f'    Optimizer: {self.optimizer_name}')
        print(f'    Learning rate: {self.lr}')
        print(f'    Weight decay: {self.weight_decay}')
        print("")

        summary(model, self.train_input_shape[1:])

        return model, criterion, optimizer

    # Training of models for a single epoch, returning the loss
    def train_one_epoch(self, train_dataloader, optimizer, criterion, model):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_dataloader)

        return average_loss
    
    # Validation of loss for validation set during training
    def validate_one_epoch(self, val_dataloader, criterion, model):
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item()

        average_loss = val_running_loss / len(val_dataloader)

        return average_loss

    # Main training loop of models
    def train(self, show_progress_plots=False, use_wandb=False, sweep=False):
        timestamp = get_current_time()
        early_stopper = EarlyStopper()

        # Initialize best model state
        best_model_state = None

        # Weights and Biases logging
        if use_wandb and wandb.run is None:
            wandb_name = self.config.current_model+"_"+timestamp if not sweep else "Sweep_"+self.config.current_model+"_"+timestamp
            # Initialize wandb run if not already done
            wandb.init(
                project="Thesis", 
                name=wandb_name,
                config={
                    "batch_size": self.batch_size,
                    "learning_rate": self.lr,
                    "weight_decay": self.weight_decay,
                    "optimizer": self.optimizer_name,
                    "dropout": self.dropout,
                    "max_epochs": self.max_epochs,
                    "loss_function": self.config.current_loss_function,
                    "model": self.config.current_model,
                    "task": self.config.task,
                    "dataset": self.config.current_dataset,
                    "freeze": self.config.models.general.freeze,
                    "pretrained": self.config.models.general.pretrained,
                    "n_resnet_blocks": self.config.models.resnet.n_blocks,
                    "tmin": self.dataset_config.tmin,
                    "tmax": self.dataset_config.tmax,
                    "channels": self.config.used_channels
            })

            if self.config.current_model == 'TCNAE':
                wandb.config.avgpool_size = self.avgpool_size

        # Setup model
        model, criterion, optimizer = self.setup_model()

        # Setup tester
        tester = Tester(test_dataloader=self.test_dataloader)

        # Choose the iterator based on show_progress_plots
        iterator = tqdm(range(self.max_epochs), desc="Training Progress") if not show_progress_plots else range(self.max_epochs)

        # Initialize test metric
        test_metric = (None, None, None)
        
        # Training loop
        for epoch in iterator:
            if epoch == 0 and show_progress_plots:
                print(f'Training model {self.config.current_model} on {self.config.current_dataset} dataset for {self.max_epochs} epochs...')
                print(model)

            # Train and validate one for epoch
            train_loss = self.train_one_epoch(self.train_dataloader, optimizer, criterion, model)
            val_loss = self.validate_one_epoch(self.val_dataloader, criterion, model)
            self.losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Test metric
            if epoch == 0 or (epoch + 1) % 100 == 0:
                test_metric = tester.test_model(current_model=model)

            # If val loss improves, save state of the model
            if epoch == 50 or (epoch > 50 and early_stopper.is_metric_improved(val_loss)):
                best_model_state = model

                # Save model as long as it isn't a sweep
                if not sweep:
                    save_model(self.config.current_dataset, self.config.current_model, self.max_epochs, timestamp, best_model_state, use_wandb=False)

            # Log metrics to wandb
            if use_wandb:
                if epoch == 0 or (epoch + 1) % 100 == 0:
                    if self.config.task == 'eeg' or 'TCNAE' in self.config.current_model:
                        wandb.log({
                            "train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "rmse": test_metric[0], 
                            "wasserstein": test_metric[1], 
                            "tss": test_metric[2],
                            "acs": test_metric[3]
                        })
                    elif self.config.task == 'class':
                        wandb.log({
                            "train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "accuracy": test_metric[0]
                        })
                        
                    wandb.run.summary["epochs"] = epoch + 1
                else:
                    if self.config.task == 'eeg' or 'TCNAE' in self.config.current_model:
                        wandb.log({
                            "train_loss": train_loss, 
                            "val_loss": val_loss, 
                        })
                    elif self.config.task == 'class':
                        wandb.log({
                            "train_loss": train_loss, 
                            "val_loss": val_loss,
                        })
                        
                    wandb.run.summary["epochs"] = epoch + 1

            # Show progress plots, if set (useful if running the job in a notebook without wandb)
            if show_progress_plots:
                print(f'Epoch {epoch+1}/{self.max_epochs} - Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

                if ((epoch + 1) % 10 == 0):
                    plot_loss(losses=self.losses, val_losses=self.val_losses)
            else:
                iterator.set_description(f"Epoch {epoch+1}/{self.max_epochs} - Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}")
            
            # Stop early if the model has converged
            if early_stopper.stop_run():
                print("Model training has not improved in", early_stopper.unimproved_count, "epochs, exceeding patience of", self.config.early_stopping.patience, ". Stopping early.")
                if not show_progress_plots:
                    iterator.close()  # Close tqdm iterator if early stopping
                break

        print("")
        print(f"Training done. Best metric: {early_stopper.best_metric:.8f}")

        # Finish the wandb run
        if use_wandb and not sweep:
            save_model(self.config.current_dataset, self.config.current_model, self.max_epochs, timestamp, best_model_state, use_wandb=use_wandb)
            wandb.finish()
            
