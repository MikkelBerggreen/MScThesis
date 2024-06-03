from utils.config_loader import ConfigLoader
from core.datahandler import DataHandler
from metrics.lossfunctions import ApproxWassersteinDistanceLoss, TimeSeriesSimilarityLoss
from metrics.performance_metrics import EEGChallengeScore
import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error
import os

# Class used to test a model either during or after training. Can evaluate with:
# - RMSE
# - Wasserstein distance
# - Time series similarity
# - A modified version of Algonauts Challenge Score
class Tester:
    def __init__(self, test_dataloader=None, changing_model_configs=None):
        self.config = ConfigLoader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if test_dataloader is None:
            if changing_model_configs is None:
                self.test_dataloader = self.setup_test_data()
            else:
                self.test_dataloader = self.setup_test_data(changing_model_configs)
        else:
            self.test_dataloader = test_dataloader

    # Setup dataloader for testing
    def setup_test_data(self, changing_model_configs=None):
        datahandler = DataHandler(changing_model_configs=changing_model_configs)
        dataloader = datahandler.get_test_loader()

        return dataloader
    
    # Run model on test dataset to get predicted values
    def get_predicted_and_actual_values(self, model, dataloader):
        all_preds = []
        all_actual = []
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            for data, targets in dataloader:
                # Ensure data is on the same device as the model
                data = data.to(self.device, dtype=torch.float)
                targets = targets.to(self.device, dtype=torch.float)

                outputs = model(data)
                all_preds.extend(outputs.cpu().numpy())
                all_actual.extend(targets.cpu().numpy())

        return np.array(all_preds), np.array(all_actual)
    
    # Evaluation metrics:
    # RMSE
    def compute_rmse(self, predicted_values, actual_values):
        actual_values = actual_values.flatten()
        predicted_values = predicted_values.flatten()
        rmse = root_mean_squared_error(actual_values, predicted_values)
        return rmse
    
    # Wasserstein distance
    def compute_wasserstein_distance(self, predicted_values, actual_values):
        predicted_values_tensor = torch.from_numpy(predicted_values).float()  # Convert to float tensor
        actual_values_tensor = torch.from_numpy(actual_values).float()

        wasserstein = ApproxWassersteinDistanceLoss()
        return wasserstein(predicted_values_tensor, actual_values_tensor)
    
    # Time series similarity
    def compute_time_series_similarity(self, predicted_values, actual_values):
        predicted_values_tensor = torch.from_numpy(predicted_values).float()
        actual_values_tensor = torch.from_numpy(actual_values).float()

        tss = TimeSeriesSimilarityLoss()
        return tss(predicted_values_tensor, actual_values_tensor)
    
    # Algonauts Challenge Score
    def compute_algonauts_challenge_score(self, predicted_values, actual_values):
        acs = EEGChallengeScore(predicted_values, actual_values)
        score_per_channel, total_score = acs.compute_scores()

        return score_per_channel, total_score
    
    # Compute accuracy (percentage) of model predictions based on predictions and actual classes in a classification task
    def compute_accuracy(self, predictions, actual):
        count = 0
        for i in range(len(predictions)):
            if predictions[i] == actual[i]:
                count += 1
        return count/len(predictions) * 100
    
    # Test trained model
    def test_model(self, model_name=None, current_model=None):
        # If we provide a model path (already trained model)
        if model_name is not None and current_model is None:
            model_directory = f'../trained_models/'
            model_path = os.path.join(model_directory, f'{model_name}.pth')
            model = torch.load(model_path, map_location=self.device)
        # If we provide a current model (in training)
        elif model_name is None and current_model is not None:
            model = current_model
        else:
            raise ValueError('Provide either only a model name (testing after training) or only a current model (testing during training)')

        predicted_values, actual_values = self.get_predicted_and_actual_values(model, self.test_dataloader)

        # If the task is regression
        if self.config.task == 'eeg' or 'TCNAE' in self.config.current_model:
            rmse = self.compute_rmse(predicted_values, actual_values)
            wasserstein = self.compute_wasserstein_distance(predicted_values, actual_values)
            # The inclusion of this metric is omitted due to its inconsistensies with the predicted response patterns
            tss = None
            _, acs = self.compute_algonauts_challenge_score(predicted_values, actual_values)
            
            return (rmse, wasserstein, tss, acs)
        # If the task is classification
        elif self.config.task == 'class':
            predicted_values = [np.argmax(label) if isinstance(label, np.ndarray) else label for label in predicted_values]
            actual_values = [np.argmax(label) if isinstance(label, np.ndarray) else label for label in actual_values]
            accuracy = self.compute_accuracy(predicted_values, actual_values)

            return (accuracy,)