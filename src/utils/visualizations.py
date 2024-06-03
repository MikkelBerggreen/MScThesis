from core.datahandler import DataHandler
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
import numpy as np
import torch
from math import floor

def plot_actual_vs_predicted_sample(model_name, sample_index=0, fig_title=None, fig_size=(12, 8), dataloader=None, channels=["O1", "Oz", "O2"], single_plot=True):
    if dataloader is None:
        data = DataHandler()
        dataloader = data.get_test_loader()
    else:
        data = dataloader

    model = torch.load(f'..\\trained_models\\{model_name}.pth', map_location=torch.device('cpu'))
    
    # Pick first sample from dataset
    for i in range(floor(sample_index / 128) + 1):
        image, eeg_data = next(iter(dataloader))
    new_index = sample_index % 128
    image_sample = image[new_index]
    eeg_sample = eeg_data[new_index]

    # Run sample through model
    if 'TCNAE' in model_name:
        input_sample = eeg_sample.unsqueeze(0)
        output_sample = model(input_sample)
    else:
        input_sample = image_sample.unsqueeze(0)
        output_sample = model(input_sample)

    # Detach output tensor and convert to numpy
    output_sample = output_sample.squeeze(0).detach().numpy()

    # Calculate RMSE between input and output
    rmse = np.sqrt(np.mean((eeg_sample.detach().numpy() - output_sample)**2, axis=1))
    print(f"RMSE: {rmse}")
    print(f"Mean RMSE: {np.mean(rmse)}")

    # Calculate wasserstein distance between input and output
    loss = SamplesLoss()
    wasserstein_distance = loss(eeg_sample, torch.tensor(output_sample))
    print(f"Wasserstein Distance: {wasserstein_distance}")

    # Visualize input vs output
    if single_plot:
        for i, channel in enumerate(channels):
            plt.figure(figsize=fig_size)
            plt.plot(eeg_sample[i], label=f'Original {channel}', alpha=0.5, color='blue', zorder=1)
            plt.plot(output_sample[i], label=f'Reconstructed {channel}', alpha=0.5, color='orange', zorder=0)
            plt.legend()
            plt.title(fig_title + f" - Channel: {channel}")
            plt.ylabel('Amplitude')
            plt.xlabel('Time (ms)')
            plt.tight_layout()
            plt.show()
    else:
        fig, axs = plt.subplots(len(channels), 1, figsize=fig_size)
        for i, channel in enumerate(channels):
            axs[i].plot(eeg_sample[i], label='Original', alpha=0.5, color='blue', zorder=1)
            axs[i].plot(output_sample[i], label='Reconstructed', alpha=0.5, color='orange', zorder=0)
            axs[i].legend()
            axs[i].set_title(f"Channel: {channel}")

        fig.suptitle(fig_title)

        for ax in axs:
            ax.set_ylabel('Amplitude')

        # Add x-axis label to last subplot
        axs[-1].set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.show()