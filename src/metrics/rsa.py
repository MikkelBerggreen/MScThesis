from utils.config_loader import ConfigLoader
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from geomloss import SamplesLoss
from tqdm import tqdm
import torch
import numpy as np

class RepresentationalSimilarityAnalysis():
    def __init__(self, data, similarity_metric='cosine'):
        self.config = ConfigLoader()
        
        self.data = data
        self.similarity_metric = similarity_metric
        self.rdm = None

        self.calculate_rdm()
    
    def distance_to_similarity(self, distance_matrix):
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is zero
        max_distance = np.max(distance_matrix)
        if max_distance == 0:
            return np.zeros_like(distance_matrix)
        similarity_matrix = distance_matrix / max_distance
        return similarity_matrix

    def approx_wasserstein_distances(self, X, Y):
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()

        # Initialize matrix
        matrix = np.empty((len(X), len(Y)))

        loss = SamplesLoss()
        for i in tqdm(range(len(X)), desc=f'Calculating Wasserstein distances'):
            x = X[i]
            for j in range(len(Y)):
                y = Y[j]
                distance = loss(x, y)
                matrix[i][j] = distance

        return matrix
    
    def rmse_distance(self, X, Y):
        matrix = np.empty((len(X), len(Y)))

        for i in tqdm(range(len(X)), desc=f'Calculating RMSE distances'):
            x = X[i]
            for j in range(len(Y)):
                y = Y[j]
                distance = np.sqrt(np.mean((x - y)**2))
                matrix[i][j] = distance

        return matrix
    
    def pearson_correlation(self, X, Y):
        matrix = np.empty((len(X), len(Y)))

        for i in tqdm(range(len(X)), desc=f'Calculating Pearson correlation'):
            x = X[i]
            for j in range(len(Y)):
                y = Y[j]
                distance = np.corrcoef(x, y)[0, 1]
                matrix[i][j] = distance

        return 1 - matrix

    def calculate_rdm(self):
        self.data = self.data.reshape(self.data.shape[0], -1)
        
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(self.data, self.data)
            self.rdm = 1 - similarity_matrix
            
        elif self.similarity_metric == 'euclidean':
            distance_matrix = euclidean_distances(self.data, self.data)
            self.rdm = self.distance_to_similarity(distance_matrix)
            
        elif self.similarity_metric == 'wasserstein':
            distance_matrix = self.approx_wasserstein_distances(self.data, self.data)
            self.rdm = self.distance_to_similarity(distance_matrix)
            
        elif self.similarity_metric == 'rmse':
            distance_matrix = self.rmse_distance(self.data, self.data)
            self.rdm = self.distance_to_similarity(distance_matrix)
            
        elif self.similarity_metric == 'pearson':
            correlation_matrix = self.pearson_correlation(self.data, self.data)
            self.rdm = correlation_matrix
            
        else:
            raise ValueError('Unknown similarity metric: {}'.format(self.similarity_metric))

    def plot_rdm(self, title=None):
        if self.rdm is None:
            self.calculate_rdm()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.rdm[:500, :500], cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()