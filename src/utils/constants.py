# Configurations
CONFIG_FILE = 'configs.yaml'
RANDOM_SEED = 42

# Data paths
RAW_EEG_FILE = 'EEG.npy.gz'
EEG_CLASSES_FILE = 'EEG_classencoding.npy.gz'
EEG_CLASSES_TESTDATA_FILE = 'EEG_classencoding_testdata.npy.gz'
EEG_LATENT_FILE = 'EEG_latent.npy.gz'
IMAGE_FILE = 'IMAGES.p'

# Trained model paths
RESNET_IMAGE_CLASSIFICATION_MODEL = 'ResNet18_50epochs_02_05_2024__13_37'
TCNAE_EEG_DECODER_MODEL = '3_channels/TCNAE_3channels'
RESNET_LATENT_EEG_MODEL = '3_channels/ResNetLatentEEG_500epochs_26_03_2024__23_55'

# Noise ceiling/floor paths
NOISE_CEILING_PATH = 'noise_ceiling.npy'
NOISE_FLOOR_PATH = 'noise_floor.npy'

# Montage, ordered channel names
CH_NAMES = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
            'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz']
