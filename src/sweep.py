from time import time
import wandb
from core.trainer import Trainer
from utils.utils import get_current_time
from utils.config_loader import ConfigLoader

class Sweep():
    def __init__(self):
        self.config = ConfigLoader()
        self.model_name = self.config.current_model
        
        self.sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'rmse',
                'goal': 'minimize'
            },
            'parameters': {
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'learning_rate': {
                    'values': [0.01, 0.001, 0.0005, 0.0001]
                    # 'min': 0.0001,  # Minimum value of the learning rate
                    # 'max': 0.01,  # Maximum value of the learning rate
                    # 'distribution': 'uniform'  # Uniform distribution
                },
                'weight_decay': {
                    'values': [0.01, 0.001, 0.0005, 0.0001]
                    # 'min': 0.0001,
                    # 'max': 0.001,
                    # 'distribution': 'uniform'
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']
                },
                'dropout': {
                    'values': [0.5, 0.6, 0.7, 0.8]
                    # 'min': 0.5,
                    # 'max': 0.85,
                    # 'distribution': 'uniform'
                },
                'max_epochs': {
                    'value': 400
                }
            }
        }

        if self.model_name == "TCNAE":
            self.sweep_config['parameters'].update({
                'avgpool_size': {
                    'values': [2, 4, 5, 10]
                }
            })

    def sweep_setup(self):
        timestamp = get_current_time()
        wandb_name = "Sweep_"+self.model_name+"_"+timestamp

        with wandb.init(name=wandb_name) as run:
            # Now you can safely access wandb.config for the current run
            batch_size = wandb.config.batch_size
            lr = wandb.config.learning_rate
            weight_decay = wandb.config.weight_decay
            optimizer_name = wandb.config.optimizer
            #dropout = wandb.config.dropout
            max_epochs = wandb.config.max_epochs

            if self.model_name == "TCNAE":
                avgpool_size = wandb.config.avgpool_size
            else:
                avgpool_size = self.config.models.tcnae.avgpool_size
            
            # Initialize the trainer with the transformed sweep config
            trainer = Trainer(batch_size=batch_size, lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name, dropout=None, max_epochs=max_epochs, avgpool_size=avgpool_size)
            trainer.train(show_progress_plots=False, use_wandb=True, sweep=True)

    def sweep_run(self):
        sweep_id = wandb.sweep(self.sweep_config, project="Thesis")
        wandb.agent(sweep_id=sweep_id, function=self.sweep_setup, project="Thesis")

if __name__ == '__main__':
    start_time = time()
    print("\nStarting sweep...")
    s = Sweep()
    s.sweep_run()
    print("Total sweep run time:", time()-start_time)