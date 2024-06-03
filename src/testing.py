
from time import time
from core.tester import Tester
from utils.config_loader import ConfigLoader
from metrics.rsa import RepresentationalSimilarityAnalysis
from pathlib import Path
import numpy as np
import scipy.stats as stats
import torch

class Run:
    def __init__(self):
        self.config = ConfigLoader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self, rsa=True):
        base_path = Path(__file__).resolve().parent

        channel_subsets = [
            ["O1", "Oz", "O2", "T7", "T8", "TP7", "TP8", "P7", "P8", "PO7", "PO8"],
            ["O1", "Oz", "O2", "T7", "T8"],
            ["O1", "Oz", "O2"]
        ]

        for subset in channel_subsets:
            print(f"Subset: {subset}")
            print('====================================')
            ch_len = len(subset)
            
            # Loop through each file in ../trained_models/{ch_len}_channels/
            models_folder = base_path / f"../trained_models/{ch_len}_channels"

            for file in (models_folder).iterdir():
                if "dropout" in file.name or "TCNAE" in file.name:
                    continue

                model_path = models_folder / file.name

                model = torch.load(model_path, map_location=self.device)
                model.eval()

                # Changing based on model name
                if 'Latent' in file.name:
                    changing_model_configs = {
                        'new_freq': 1000,
                        'batch_size': 256,
                        'used_channels': subset
                    }
                elif 'Baseline' in file.name:
                    changing_model_configs = {
                        'new_freq': 100,
                        'batch_size': 128,
                        'used_channels': subset
                    }
                else:
                    changing_model_configs = {
                        'new_freq': 100,
                        'batch_size': 128,
                        'used_channels': subset
                    }

                # Load data
                tester = Tester(changing_model_configs=changing_model_configs)

                if rsa:
                    reconstructed, target = tester.get_predicted_and_actual_values(model, tester.test_dataloader)

                    # Compute metrics
                    rsa_target = RepresentationalSimilarityAnalysis(target, similarity_metric='rmse')
                    rsa_reconstructed = RepresentationalSimilarityAnalysis(reconstructed, similarity_metric='rmse')

                    def flatten_upper_triangular(matrix):
                        return matrix[np.triu_indices_from(matrix, k=1)]

                    flattened_rsa1 = flatten_upper_triangular(rsa_target.rdm)
                    flattened_rsa2 = flatten_upper_triangular(rsa_reconstructed.rdm)

                    # Pearson Correlation
                    pearson_corr, pearson_p_value = stats.pearsonr(flattened_rsa1, flattened_rsa2)
                    print(f'== {file.name} - Pearson Correlation: {pearson_corr}, p-value: {pearson_p_value}')
                else:
                    rmse, wasserstein, _, acs = tester.test_model(current_model=model)
                    print(f'== {file.name} - {rmse} & {wasserstein} & {acs}')
                    print('')

if __name__ == '__main__':
    start_time = time()
    print("\nStarting run...")
    r = Run()
    r.run(rsa=False)
    print("Total run time:", time() - start_time)