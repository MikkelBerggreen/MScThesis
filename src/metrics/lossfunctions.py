import torch
import torch.nn as nn
from utils.config_loader import ConfigLoader
from geomloss import SamplesLoss

# This class is used to measure similarity between time series as a loss function.
# The method takes inspiration from the paper:
# "A New Time Series Similarity Measurement Method Based on Fluctuation Features"
# by Hailan CHEN, Xuedong GAO
# Link: https://hrcak.srce.hr/file/351984

# By using fluctuation features of the time-series data, we can better attempt to combat time-shifts in the EEG data.
# The function is quite costly in its computations, so it's most likely more useful as an after training evaluation tool.

class TimeSeriesSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ConfigLoader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def identify_extreme_points(self, outputs, targets):
        def vectorized_identify_extreme_points_with_values(series):
            series_tensor = torch.tensor(series, device=self.device, dtype=torch.float32)
            n = len(series_tensor)
            extreme_points = torch.zeros(n, device=self.device, dtype=torch.int64)  # Adjusted to match torch.where output dtype
            
            # Identifying extreme points
            left_diff = series_tensor[1:-1] - series_tensor[:-2]
            right_diff = series_tensor[1:-1] - series_tensor[2:]
            condition = left_diff * right_diff > 0
            extreme_points[1:-1][condition] = torch.where(left_diff[condition] > 0, torch.tensor(1, device=self.device, dtype=torch.int64), torch.tensor(-1, device=self.device, dtype=torch.int64))
            
            # Handling first and last points separately
            extreme_points[0] = 1 if series_tensor[0] > series_tensor[1] else -1
            extreme_points[-1] = 1 if series_tensor[-1] > series_tensor[-2] else -1
            
            # Combining series values and extreme points into a 2D tensor
            combined_tensor = torch.stack([series_tensor, extreme_points.to(torch.float32)], dim=1)
            
            return combined_tensor

        # Re-test the function with the adjustment for combining values and indicators
        outputs_extreme_points_with_values = vectorized_identify_extreme_points_with_values(outputs)
        targets_extreme_points_with_values = vectorized_identify_extreme_points_with_values(targets)

        return outputs_extreme_points_with_values, targets_extreme_points_with_values

    def identify_candidate_fluctuation_points(self, outputs_extreme_points_with_values, targets_extreme_points_with_values, candidate_threshold):
        def vectorized_identify_candidate_fluctuation_points(extreme_points_tensor):
            # Assume extreme_points_tensor is a 2D tensor where the first column is the series values
            # and the second column is the extreme point indicator (1 or -1 for extreme points).

            # Initialize a tensor to hold candidate fluctuation point indicators, default to False (0)
            n = extreme_points_tensor.size(0)
            candidate_fluctuation_points = torch.zeros(n, device=self.device, dtype=torch.int64)

            # Find indices of extreme points
            extreme_indices = torch.where(extreme_points_tensor[:, 1] != 0)[0]

            # Starting point is considered a candidate fluctuation point
            candidate_fluctuation_points[extreme_indices[0]] = 1

            # Calculate differences between consecutive extreme points
            for i in range(1, len(extreme_indices)):
                current_value = extreme_points_tensor[extreme_indices[i], 0]
                previous_value = extreme_points_tensor[extreme_indices[i-1], 0]
                if abs(current_value - previous_value) > candidate_threshold:
                    candidate_fluctuation_points[extreme_indices[i]] = 1

            # Combine the original tensor with the new candidate fluctuation points indicators
            combined_tensor = torch.cat([extreme_points_tensor, candidate_fluctuation_points.unsqueeze(1).to(torch.float32)], dim=1)
            
            return combined_tensor

        # Assume outputs_extreme_points_with_values and targets_extreme_points_with_values are the input tensors from the previous step
        outputs_candidate_fluctuation_points = vectorized_identify_candidate_fluctuation_points(outputs_extreme_points_with_values)
        targets_candidate_fluctuation_points = vectorized_identify_candidate_fluctuation_points(targets_extreme_points_with_values)

        return outputs_candidate_fluctuation_points, targets_candidate_fluctuation_points

    def determine_fluctuation_points(self, outputs_candidate_fluctuation_points, targets_candidate_fluctuation_points):
        def vectorized_determine_fluctuation_points_corrected(candidate_fluctuation_tensor):
            # Assuming candidate_fluctuation_tensor structure is:
            # Column 0: Series values, Column 1: Extreme point indicators, Column 2: Candidate fluctuation indicators
            
            n = candidate_fluctuation_tensor.size(0)
            fluctuation_points = torch.zeros(n, device=self.device, dtype=torch.int64)
            
            # Always consider the starting point as a fluctuation point
            fluctuation_points[0] = 1

            # Iterate over candidate fluctuation points to dynamically adjust based on sequence
            for i in range(1, n):
                if candidate_fluctuation_tensor[i, 2] == 1:  # If it's a candidate fluctuation point
                    prev_i = i - 1
                    # Find the previous candidate fluctuation point
                    while prev_i >= 0 and candidate_fluctuation_tensor[prev_i, 2] == 0:
                        prev_i -= 1
                    
                    # If no previous candidate fluctuation point found, continue
                    if prev_i < 0:
                        continue

                    current_attr = candidate_fluctuation_tensor[i, 1]
                    prev_attr = candidate_fluctuation_tensor[prev_i, 1]

                    # Check attributes for current and previous candidate points
                    if current_attr * prev_attr == -1:  # Opposite attributes
                        fluctuation_points[i] = 1
                    else:  # Same attributes, decide which to keep based on values
                        if (current_attr == 1 and candidate_fluctuation_tensor[i, 0] > candidate_fluctuation_tensor[prev_i, 0]) or \
                        (current_attr == -1 and candidate_fluctuation_tensor[i, 0] < candidate_fluctuation_tensor[prev_i, 0]):
                            fluctuation_points[i] = 1
                            fluctuation_points[prev_i] = 0  # Update previous point status
                        else:
                            fluctuation_points[prev_i] = 1  # Retain previous point, no need to update current point

            # Combine the original tensor with the new fluctuation points indicators
            combined_tensor = torch.cat([candidate_fluctuation_tensor, fluctuation_points.unsqueeze(1).to(torch.float32)], dim=1)
            
            return combined_tensor

        # Corrected usage
        outputs_fluctuation_points_corrected = vectorized_determine_fluctuation_points_corrected(outputs_candidate_fluctuation_points)
        targets_fluctuation_points_corrected = vectorized_determine_fluctuation_points_corrected(targets_candidate_fluctuation_points)

        return outputs_fluctuation_points_corrected, targets_fluctuation_points_corrected
    
    def similarity_matching(self, X, Y, t):
        matched_pairs = []  # To store matched index pairs (i, j)

        # Keep track of indices in Y that have been matched to avoid duplicates
        matched_indices_Y = set()

        # Iterate through each point in X
        for i, x in enumerate(X):
            if not x[3]:  # Skip if Fluctuation is False
                continue
            
            best_match_j = None
            min_distance = float('inf')
            
            # Iterate through each point in Y within the threshold range around i
            for j in range(max(0, i - t), min(len(Y), i + t + 1)):
                y = Y[j]
                if not y[3] or j in matched_indices_Y:  # Skip if Fluctuation is False or already matched
                    continue
                
                if x[1] == y[1]:  # Check if attributes match
                    distance = abs(i - j)
                    if distance < min_distance:
                        best_match_j = j
                        min_distance = distance
            
            # If a match is found, add it to the matched pairs and mark the index in Y as matched
            if best_match_j is not None:
                matched_pairs.append((i, best_match_j))
                matched_indices_Y.add(best_match_j)

        return matched_pairs

    def similarity_matching_sorted(self, output_fluctuations, target_fluctuations, threshold):
        # List to hold all potential matches (i, j, distance)
        potential_matches = []

        # Identify all potential matches
        for i, x in enumerate(output_fluctuations):
            if not x[3]:  # Skip if Fluctuation is False for x
                continue

            for j, y in enumerate(target_fluctuations):
                if not y[3]:  # Skip if Fluctuation is False for y
                    continue

                # Check if within threshold and attributes match
                if abs(i - j) <= threshold and x[1] == y[1]:
                    potential_matches.append((i, j, abs(i - j)))

        # Sort potential matches by distance (third element of each tuple)
        potential_matches.sort(key=lambda match: match[2])

        matched_pairs = []  # To store the final matched pairs
        matched_in_X = set()
        matched_in_Y = set()

        # Select matches, prioritizing closest matches first
        for i, j, _ in potential_matches:
            if i not in matched_in_X and j not in matched_in_Y:
                matched_pairs.append((i, j))
                matched_in_X.add(i)
                matched_in_Y.add(j)

        return matched_pairs

    def similarity_matching_dp(self, output_fluctuations, target_fluctuations, threshold):
        lenX, lenY = len(output_fluctuations), len(target_fluctuations)
        dp = [[0] * (lenY + 1) for _ in range(lenX + 1)]
        path = [[''] * (lenY + 1) for _ in range(lenX + 1)]

        # Step 1: DP Table Construction
        for i in range(1, lenX + 1):
            for j in range(1, lenY + 1):
                if output_fluctuations[i-1][3] and target_fluctuations[j-1][3] and output_fluctuations[i-1][1] == target_fluctuations[j-1][1] and abs(i - j) <= threshold:
                    dp[i][j] = dp[i-1][j-1] + 1
                    path[i][j] = 'match'
                else:
                    if dp[i-1][j] > dp[i][j-1]:
                        dp[i][j] = dp[i-1][j]
                        path[i][j] = 'up'
                    else:
                        dp[i][j] = dp[i][j-1]
                        path[i][j] = 'left'

        # Step 2: Traceback with Proximity Preference
        matches = []
        i, j = lenX, lenY
        while i > 0 and j > 0:
            if path[i][j] == 'match':
                # Prioritize closer matches within the threshold
                matches.append((i-1, j-1))  # Adding the match
                i, j = i-1, j-1  # Move diagonally back for a match
            elif path[i][j] == 'up':
                i -= 1
            elif path[i][j] == 'left':
                j -= 1

        matches.reverse()  # Reverse to get the correct order
        return matches

    def get_fluctuation_point_indices(self, series):
        return [i for i, x in enumerate(series) if x[3]]

    def compute_distance(self, output_fluctuations, target_fluctuations, similarity_matches):
        def distances_of_fluctuation_matches():
            # Convert similarity matches to tensor for efficient indexing
            similarity_matches_tensor = torch.tensor(similarity_matches, device=self.device, dtype=torch.long)
            
            # Index output and target fluctuations using the similarity matches
            output_values = output_fluctuations[similarity_matches_tensor[:, 0], 0]
            target_values = target_fluctuations[similarity_matches_tensor[:, 1], 0]
            
            # Compute the squared distances
            squared_distances = torch.pow(output_values - target_values, 2)
            
            # Sum the squared distances
            sum_squared_distances = torch.sum(squared_distances)
            
            return sum_squared_distances

        def fluctuation_degree():
            def compute_fluctuation_degree(series):
                # Get fluctuation point indices
                fluctuation_points_indices = self.get_fluctuation_point_indices(series)

                # Extract the series values at fluctuation points
                fluctuation_values = series[fluctuation_points_indices, 0]

                # Calculate differences between adjacent fluctuation points
                differences = torch.abs(fluctuation_values[1:] - fluctuation_values[:-1])

                # Initialize fluctuation degrees with zeros
                fluctuation_degrees = torch.zeros_like(series[:, 0], device=self.device)

                # Assign the average of adjacent differences to each fluctuation point
                # For the first and last fluctuation point, use the direct difference
                fluctuation_degrees[fluctuation_points_indices[1:-1]] = (differences[:-1] + differences[1:]) / 2
                fluctuation_degrees[fluctuation_points_indices[0]] = differences[0]
                fluctuation_degrees[fluctuation_points_indices[-1]] = differences[-1]

                return fluctuation_degrees
            
            output_fluctuation_degrees = compute_fluctuation_degree(output_fluctuations)
            target_fluctuation_degrees = compute_fluctuation_degree(target_fluctuations)

            return output_fluctuation_degrees, target_fluctuation_degrees

        def information_weight(output_fluctuation_degrees, target_fluctuation_degrees):
            def compute_information_weight(fluctuation_degrees):
                # Calculate the total sum of fluctuation degrees
                total_fluctuation_degree = torch.sum(fluctuation_degrees)

                # Calculate information weights for each fluctuation point
                information_weights = fluctuation_degrees / total_fluctuation_degree

                return information_weights
                        
            output_information_weights = compute_information_weight(output_fluctuation_degrees)
            target_information_weights = compute_information_weight(target_fluctuation_degrees)

            return output_information_weights, target_information_weights
                    
        def similarity_matching_degree(output_information_weights, target_information_weights):
            output_information_weights_sum = sum([x for x in output_information_weights if x is not None])
            target_information_weights_sum = sum([x for x in target_information_weights if x is not None])

            smd = min(output_information_weights_sum, target_information_weights_sum)

            return smd
        
        # Compute distance:
        output_fd, target_fd = fluctuation_degree()
        output_iw, target_iw = information_weight(output_fd, target_fd)
        smd = similarity_matching_degree(output_iw, target_iw)
        match_distances = distances_of_fluctuation_matches()
        distance = 1/smd * match_distances
        
        return distance

    def forward(self, outputs, targets):
        n_batches = len(outputs)
        n_channels = len(outputs[0])
        distances_batch_sum = 0
        for batch in range(n_batches):
            acc_distances = 0
            # For each channel, compute the similarity of the outputs and targets for that channel's time-series
            for channel in range(n_channels):
                # Step 2.1
                outputs_extreme_points, targets_extreme_points = self.identify_extreme_points(outputs[batch][channel], targets[batch][channel])
                # Step 2.2
                candidate_threshold = 0.1 * 1e-6 * self.config.transformations.eeg.scaling
                outputs_candidate_fluctuation_points, targets_candidate_fluctuation_points = self.identify_candidate_fluctuation_points(outputs_extreme_points, targets_extreme_points, candidate_threshold)
                # Step 2.3
                outputs_fluctuation_points, targets_fluctuation_points = self.determine_fluctuation_points(outputs_candidate_fluctuation_points, targets_candidate_fluctuation_points)
                # Step 3.1
                similarity_match_threshold = 10
                similarity_matches = self.similarity_matching(outputs_fluctuation_points, targets_fluctuation_points, similarity_match_threshold)
                # Step 3.2
                distance = self.compute_distance(outputs_fluctuation_points, targets_fluctuation_points, similarity_matches)

                acc_distances += torch.pow(distance, 2)

            rmse_distances = torch.sqrt(acc_distances / n_channels)
            distances_batch_sum += rmse_distances

        mean_distances_batch_sum = distances_batch_sum / n_batches

        return mean_distances_batch_sum

# This class is used to approximate the Wasserstein distance as a loss function.
# As with the fluctuation features of the time-series data, this method also attempts to better combat time-shifts in the EEG data.
# This loss function is less costly in its computations, so it's most likely more useful during training.
class ApproxWassersteinDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = SamplesLoss()
    
    def forward(self, outputs, targets):
        n_channels = len(outputs[0])
        acc_distances = 0
        # For each channel, compute the Wasserstein distance of the outputs and targets for that channel's time-series
        for channel in range(n_channels):
            distance = self.loss(outputs[channel], targets[channel])
            acc_distances += torch.pow(distance, 2)

        rmse_distances = torch.sqrt(acc_distances / n_channels)

        return rmse_distances