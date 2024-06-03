from core.datahandler import DataHandler
from scipy.stats import pearsonr
import numpy as np
import torch
from scipy.stats import gmean
from tqdm import tqdm

class NoiseCeiling():
    def __init__(self, eeg_data=None, image_data=None):
        # Initialize DataHandler only if needed
        if eeg_data is None or image_data is None:
            data_handler = DataHandler()
            self.eeg_data = eeg_data if eeg_data is not None else data_handler.eeg_data
            self.image_data = image_data if image_data is not None else data_handler.image_data
        else:
            self.eeg_data = eeg_data
            self.image_data = image_data

        # Shape (N, C, T) where N is number of samples, C is number of channels, T is number of timepoints
        self.channels = self.eeg_data.shape[1]
        self.timepoints = self.eeg_data.shape[2]

        self.sorted_eeg = self.prepare_data()
    
    # Compute noise ceiling for each channel
    def compute_noise_ceiling(self, data1, data2):
        noise_ceilings = np.zeros(self.channels)

        for channel in range(self.channels):
            channel_correlations = []
            for timepoint in range(self.timepoints):
                # Correlation for each timepoint in this channel
                data1_timepoint = data1[:, channel, timepoint]
                data2_timepoint = data2[:, channel, timepoint]

                correlation, _ = pearsonr(data1_timepoint, data2_timepoint)
                if not np.isnan(correlation):  # Check for nan which occurs if data has no variance
                    # Set negative correlations to 0 (not sure if this is necessary)
                    #if correlation < 0:
                    #    correlation = 0
                    channel_correlations.append(correlation)


            # Average correlation across all timepoints for this channel, if any correlations were computed
            if channel_correlations:
                noise_ceilings[channel] = np.mean(channel_correlations)
            else:
                noise_ceilings[channel] = np.nan  # Handle case where all correlations are nan

        return noise_ceilings

    # Compute noise floor for each channel
    def compute_robust_noise_floor(self, n_permutations=500):
        noise_floor = np.zeros((n_permutations, self.channels))
        indices = torch.randperm(len(self.eeg_data))

        # tqdm iterator setup
        iterator = tqdm(range(n_permutations), desc="Computing noise floor")

        for idx in iterator:
            midpoint = self.eeg_data.shape[0] // 2
            data1_indices = indices[:midpoint]
            data2_indices = indices[midpoint:]

            data1 = self.eeg_data[data1_indices]
            data2 = self.eeg_data[data2_indices]

            channel_noise_floor = self.compute_noise_ceiling(data1, data2)
            noise_floor[idx] = np.abs(channel_noise_floor)

        # Compute noise floor across classes (1854, 11)
        # geometric_mean_noise_floor = gmean(noise_floor, axis=0, nan_policy='omit')
        mean_noise_floor = np.mean(noise_floor, axis=0)

        print("Robust noise floor: ", mean_noise_floor)
        return mean_noise_floor
    
    # Compute robust noise ceiling
    def compute_robust_noise_ceiling(self, n_permutations=50):
        noise_ceilings = np.zeros((len(self.sorted_eeg) // 12, self.channels))

        # tqdm iterator setup
        iterator = tqdm(range(0, len(self.sorted_eeg), 12), desc="Computing noise ceiling")

        for idx in iterator:
            current_eeg = np.array(self.sorted_eeg[idx:idx+12])

            class_noise_ceilings = np.zeros(self.channels)

            for _ in range(n_permutations):
                indices = torch.randperm(len(current_eeg)).numpy()

                data1_indices = indices[:6]
                data2_indices = indices[6:]

                data1 = current_eeg[data1_indices]
                data2 = current_eeg[data2_indices]

                channel_noise_ceilings = self.compute_noise_ceiling(data1, data2)
                class_noise_ceilings += np.abs(channel_noise_ceilings)

            class_noise_ceilings /= n_permutations
            noise_ceilings[idx//12] = class_noise_ceilings

        # Compute noise ceilings across classes (1854, 11)
        # geometric_mean_noise_ceilings = gmean(noise_ceilings, axis=0, nan_policy='omit')
        mean_noise_ceilings = np.mean(noise_ceilings, axis=0)

        print("Robust noise ceiling: ", mean_noise_ceilings)
        return mean_noise_ceilings
    
    def prepare_data(self):
        zipped = zip(self.image_data, self.eeg_data)
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        _, sorted_eeg = zip(*sorted_zipped)

        return np.array(sorted_eeg)
