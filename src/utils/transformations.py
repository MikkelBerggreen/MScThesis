import torchvision.transforms as TV
import torchaudio.transforms as TA
from utils.config_loader import ConfigLoader
from utils.constants import CH_NAMES, TCNAE_EEG_DECODER_MODEL
from scipy import signal
from sklearn.preprocessing import RobustScaler
import torch
import gzip
import numpy as np
from utils.model_setup import get_model_from_name
from pathlib import Path
from utils.constants import EEG_LATENT_FILE

# Transform images based on model
def transform_image(data, config):
    model_name = config.current_model 
    dimensions = config.transformations.image.dimensions
    
    # Image transformation mappings dictionary containing transformations to use for each model
    # (currently they are all identical, but can be individually changed based on which model is experimented with)
    image_transforms_mapping = {
        'BaselineEEGReconstruction': TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'AlexNet': TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'SqueezeNet': TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'ResNet': TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'ResNetLatentEEG': TV.Compose([
            TV.Resize(256),
            TV.CenterCrop(dimensions),
            TV.ToTensor(),
            TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Add new models and their transformations here
    }

    transform = image_transforms_mapping[model_name] or TV.ToTensor()
    return transform(data)

def z_score_normalize(data):
    n_samples, n_channels, n_timepoints = data.shape
    normalized_data = torch.empty((n_samples, n_channels, n_timepoints))

    # Do normalization across each channel separately
    for channel in range(n_channels):
        channel_data = data[:, channel, :]
        channel_mean = torch.mean(channel_data, axis=0)
        channel_std = torch.std(channel_data, axis=0)
        normalized_channel_data = (channel_data - channel_mean) / channel_std
        normalized_data[:, channel, :] = normalized_channel_data

    return normalized_data

# Function used to transform EEG data to latent space, saving it in a file.
def eeg_to_latent(eeg_data):
    config = ConfigLoader()
    base_path = Path(__file__).resolve().parent
    save_path = (base_path / f"../../data/datasets/{config.current_dataset}/{EEG_LATENT_FILE}").resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_path.exists():
        return
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load TCNAE model
    model_path = (base_path / f"../../trained_models/{TCNAE_EEG_DECODER_MODEL}.pth").resolve()
    tcnae_model = torch.load(model_path, map_location=device)

    tcnaeencoder = torch.nn.Sequential(*list(tcnae_model.children())[:16]) # First 16 layers are the encoder
    tcnaeencoder = tcnaeencoder.to(device)
    tcnaeencoder.eval()

    latent_eeg = tcnaeencoder(eeg_data)
    latent_eeg = latent_eeg.detach().numpy()

    with gzip.open(f"{save_path}", 'wb') as f:
        np.save(f, latent_eeg)

# Transform EEG data
# - Take only specified channels (from config)
# - Filter, resample, baseline correct, epoch, and robust scale the EEG data as needed
def transform_eeg(data, changing_model_configs=None):
    config = ConfigLoader()
    dataset_name = config.current_dataset
    dataset_config = getattr(config.datasets, dataset_name)
    if changing_model_configs is None:
        frequency_scalar = config.transformations.eeg.new_freq / 1000
    else:
        frequency_scalar = changing_model_configs['new_freq'] / 1000
    t_min = int(dataset_config.tmin * frequency_scalar)
    t_max = int(dataset_config.tmax * frequency_scalar)

    # Remove unwanted channels from EEG data
    if changing_model_configs is None:
        channels = config.used_channels
    else:
        channels = changing_model_configs['used_channels']
    if channels != 'All':
        channel_indices = []
        for channel in channels:
            if channel in CH_NAMES:
                idx = CH_NAMES.index(channel)
                channel_indices.append(idx)
        data = data[:, channel_indices] 

    # Bandpass filter data
    original_frequency = 1000
    low_pass = config.transformations.eeg.low_pass
    high_pass = config.transformations.eeg.high_pass
    order = 4
    nyquist = 0.5 * original_frequency
    low_pass_norm = low_pass / nyquist
    high_pass_norm = high_pass / nyquist
    sos = signal.butter(order, [high_pass_norm, low_pass_norm], btype='bandpass', output='sos')
    data = np.copy(signal.sosfiltfilt(sos, data))

    # Downsample data (happens after filtering, since TA doesn't seem to do much filtering itself)
    data = torch.tensor(data, dtype=torch.float)
    if changing_model_configs is None:
        transform = TA.Resample(
        orig_freq=1000,
        new_freq=config.transformations.eeg.new_freq
        )
    else:
        transform = TA.Resample(
        orig_freq=1000,
        new_freq=changing_model_configs['new_freq']
        )
    data = transform(data)

    # Baseline correction (and cut to specified interval)
    baseline_correction = torch.mean(data[:, :, 0:t_min], axis=2).unsqueeze(-1)
    data = (data - baseline_correction)[:, :, t_min:t_max]

    # Channel wise Robust Scalar
    for channel in range(len(channels)):
        data[:, channel, :] = torch.from_numpy(RobustScaler().fit_transform(data[:, channel, :]))

    # other transformations ...

    print("Done transforming EEG data.")
    return data