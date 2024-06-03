from utils.config_loader import ConfigLoader

# Class used to stop model training early if it tends to converge
class EarlyStopper:
    def __init__(self):
        self.config = ConfigLoader()

        self.best_metric = float('inf') if self.config.task == 'eeg' else float('-inf')
        self.unimproved_count = 0

    # Return True if patience has been exceeded, False otherwise
    def stop_run(self):
        return self.unimproved_count > self.config.early_stopping.patience

    # Checks if a provided metric value has improved by a set threshold in configs based on previously provided metric values
    def is_metric_improved(self, new_metric):
        if self.config.task == 'class':
            if new_metric > (self.best_metric + self.config.early_stopping.min_change):
                self.best_metric = new_metric
                self.unimproved_count = 0
                return True
            
            self.unimproved_count += 1
            return False
        elif self.config.task == 'eeg':
            if new_metric < (self.best_metric - self.config.early_stopping.min_change):
                self.best_metric = new_metric
                self.unimproved_count = 0
                return True

            self.unimproved_count += 1
            return False
        else:
            raise ValueError("Task not recognized. Please check the task in the configs.")