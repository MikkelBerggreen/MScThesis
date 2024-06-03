from scipy.stats import pearsonr
import torch
import torch.nn as nn
from utils.config_loader import ConfigLoader
import numpy as np
from pathlib import Path
from utils.constants import NOISE_CEILING_PATH, NOISE_FLOOR_PATH

class AlgonautsChallengeScore():
    def __init__(self, predicted, actual):
        self.config = ConfigLoader()
        self.dataset_name = self.config.current_dataset
        self.model = self.config.current_model

        self.predicted = predicted
        self.actual = actual

        # Load noise ceiling
        base_path = Path(__file__).resolve().parent
        if self.model == "ResNetLatentEEG" or self.model == "TCNAE":
            noise_ceiling_file = NOISE_CEILING_PATH+"_tcnae"
        else:
            noise_ceiling_file = NOISE_CEILING_PATH
        noise_ceiling_path = (base_path / f"../../data/datasets/{self.dataset_name}/{noise_ceiling_file}").resolve()
        self.noise_ceiling = np.load(noise_ceiling_path)

    def compute_channel_pearsons(self, predicted, actual):
        channel_pearsons = np.zeros(predicted.shape[1:])
        for channel in range(len(predicted[0])):
            mean_predicted = np.mean(predicted[:, channel], axis=0)
            mean_actual = np.mean(actual[:, channel], axis=0)
            numerator = np.sum((actual[:, channel] - mean_actual) * (predicted[:, channel] - mean_predicted), axis=0)
            denominator1 = np.sum((actual[:, channel] - mean_actual)**2, axis=0)
            denominator2 = np.sum((predicted[:, channel] - mean_predicted)**2, axis=0)
            denominator = np.sqrt(denominator1 * denominator2)

            pearson = numerator / denominator
            channel_pearsons[channel] = pearson
            
        return channel_pearsons

    def compute_channel_challenge_scores(self):
        channel_pearsons = self.compute_channel_pearsons(self.predicted, self.actual)
        channels = len(channel_pearsons)

        print(channel_pearsons.shape)

        # Compute a score for each channel because we have an extra dimension
        channel_challenge_scores_per_channel = np.mean(channel_pearsons, axis=1)
        print(channel_challenge_scores_per_channel.shape)

        # Compute a score for all channels combined
        channel_challenge_scores_total = np.mean(channel_challenge_scores_per_channel ** 2) * 100 #/ self.noise_ceiling[:channels]) * 100
        print(channel_challenge_scores_total.shape)

        return channel_challenge_scores_per_channel, channel_challenge_scores_total

class EEGChallengeScore():
    def __init__(self, predicted, actual):
        self.config = ConfigLoader()
        self.dataset_name = self.config.current_dataset
        self.model = self.config.current_model

        self.predicted = predicted
        self.actual = actual

        # Load noise ceiling
        # base_path = Path(__file__).resolve().parent
        # if self.model == "ResNetLatentEEG" or self.model == "TCNAE":
        #     noise_ceiling_file = NOISE_CEILING_PATH+"_tcnae"
        # else:
        #     noise_ceiling_file = NOISE_CEILING_PATH
        # noise_ceiling_path = (base_path / f"../../data/datasets/{self.dataset_name}/{noise_ceiling_file}").resolve()
        # self.noise_ceiling = np.load(noise_ceiling_path)

    def compute_channel_pearsons(self):
        # Number of channels is the second dimension
        n_channels = self.predicted.shape[1]
        pearson_coeffs = np.zeros(n_channels)

        for channel in range(n_channels):
            for timepoint in range(self.predicted.shape[2]):
                r, _ = pearsonr(self.actual[:, channel, timepoint], self.predicted[:, channel, timepoint])
                pearson_coeffs[channel] += r

            pearson_coeffs[channel] /= self.predicted.shape[2]  # Average across timepoints if needed

        return pearson_coeffs

    def compute_scores(self):
        pearson_coeffs = self.compute_channel_pearsons()
        scores = pearson_coeffs #/ self.noise_ceiling[:len(pearson_coeffs)]
        total_score = np.mean(scores) * 100  # scale to 0-100%
        return scores, total_score