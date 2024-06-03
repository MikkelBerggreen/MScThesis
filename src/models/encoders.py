from utils.config_loader import ConfigLoader
from utils.constants import RESNET_IMAGE_CLASSIFICATION_MODEL, TCNAE_EEG_DECODER_MODEL
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

import torch 
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet152_Weights, AlexNet_Weights, SqueezeNet1_0_Weights

# -------- Base Encoder class -------- #

class Encoder(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.config = ConfigLoader()
        self.dataset_name = self.config.current_dataset
        self.dataset_config = getattr(self.config.datasets, self.dataset_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        raise NotImplementedError
    
    def compute_output_shape(self, conv_layers_list, shape, flatten=True):
        batch_size = self.dataset_config.batch_size
        
        input = torch.autograd.Variable(torch.rand(batch_size, *shape)).to(self.device)
        # Sequentially apply each conv layer block if conv_layers is iterable
        if isinstance(conv_layers_list, nn.Sequential):
            for conv_layers in conv_layers_list:
                conv_layers.to(self.device)
                input = conv_layers(input)
        else:
            conv_layers_list.to(self.device)
            input = conv_layers_list(input)

        if flatten:
            n_size = input.data.view(batch_size, -1).size(1)
            return n_size
        
        return input.data.shape
    
# -------- Specific Encoders -------- #

class BaselineEncoder(Encoder):
    def __init__(self, input_shape, dropout=None):
        super().__init__(input_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout

        self.mlp_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((28, 28)),
            nn.Flatten(),
            nn.Linear(input_shape[1] * 28 * 28, 1000),
            #nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            #nn.Dropout(self.dropout),
            nn.ReLU(),
        )

        dimensions = self.config.transformations.image.dimensions
        self.encoder_output = self.compute_output_shape(self.mlp_layers, (3, dimensions, dimensions))
    
    def forward(self, x):
        x = self.mlp_layers(x)
        return x
    
class AlexNetEncoder(Encoder):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        base_model = models.alexnet(weights=AlexNet_Weights.DEFAULT if self.config.models.general.pretrained else None)
        self.conv_layers = base_model.features
        
        dimensions = self.config.transformations.image.dimensions
        self.encoder_output = self.compute_output_shape(self.conv_layers, (3, dimensions, dimensions))
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x

class SqueezeNetEncoder(Encoder):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        base_model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT if self.config.models.general.pretrained else None)
        self.conv_layers = base_model.features

        dimensions = self.config.transformations.image.dimensions
        self.encoder_output = self.compute_output_shape(self.conv_layers, (3, dimensions, dimensions))

    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
class ResNetEncoder(Encoder):
    def __init__(self, input_shape, n_blocks=4):
        super().__init__(input_shape)

        if self.config.models.general.pretrained:
            base_path = Path(__file__).resolve().parent
            resnet_path = (base_path / f"../../trained_models/{RESNET_IMAGE_CLASSIFICATION_MODEL}.pth").resolve()
            base_model = torch.load(resnet_path, map_location=self.device)
            print("Base model:", base_model)
            self.conv_layers = base_model.encoder.conv_layers
            self.avgpool = base_model.encoder.avgpool
        else:
            base_model = models.resnet18(weights=None)
            print("Base model:", base_model)
        
            self.conv_layers = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.avgpool = base_model.avgpool

        # Setup conv layers based on number of blocks
        basicblock_index = 4 + n_blocks
        self.conv_layers = self.conv_layers[0:basicblock_index] # 5 to 8

        dimensions = self.config.transformations.image.dimensions
        conv_output_shape = self.compute_output_shape(self.conv_layers, (3, dimensions, dimensions), flatten=False)
        self.encoder_output = self.compute_output_shape(self.avgpool, conv_output_shape[1:])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        return x
    
class PretrainedResNetLatentEncoder(Encoder):
    def __init__(self, input_shape, dropout=None, n_blocks=4):
        super().__init__(input_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout
        tmin = self.dataset_config.tmin
        tmax = self.dataset_config.tmax
        frequency = self.config.transformations.eeg.new_freq
        dimensions = self.config.transformations.image.dimensions

        # Load ResNet model
        base_path = Path(__file__).resolve().parent
        resnet_path = (base_path / f"../../trained_models/{RESNET_IMAGE_CLASSIFICATION_MODEL}.pth").resolve()
        base_model = torch.load(resnet_path, map_location=self.device)

        # Load TCNAE model to get avg pool size
        model_path = (base_path / f"../../trained_models/{TCNAE_EEG_DECODER_MODEL}.pth").resolve()
        tcnae_model = torch.load(model_path, map_location=self.device)

        self.avgpool_size = tcnae_model.avgpool_size
        self.latent_timepoints = int((((tmax - tmin) / 1000) * frequency) / self.avgpool_size)

        # Setup conv layers based on number of blocks
        basicblock_index = 4 + n_blocks
        self.conv_layers = base_model.encoder.conv_layers[0:basicblock_index] # 5 to 8

        # Set up avgpool layer with dynamic adpative avgpool size
        conv_output_dims = self.compute_output_shape(self.conv_layers, (3, dimensions, dimensions), flatten=False)
        adaptive_avgpool_size = int(round((512 / conv_output_dims[1]) ** 0.5)) # Match output size as close to original ResNet output size as possible ((512, 1, 1) = 512)
        self.avgpool = nn.AdaptiveAvgPool2d((adaptive_avgpool_size, adaptive_avgpool_size))
        
        # Set avgpool to match ResNet standard
        #self.avgpool = base_model.decoder.avgpool
        
        # Calculate ResNet output size after avgpool
        self.encoder_output = self.compute_output_shape(self.avgpool, conv_output_dims[1:])
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.encoder_output, out_features=256),
            nn.Dropout(self.dropout),    
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(self.dropout),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 4 * self.latent_timepoints)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = x.view(-1, 4, self.latent_timepoints) # first dimension is 4 because the TCNAE encoder compresses data into 4 channels
        return x

class RWResNetLatentEncoder(Encoder):
    def __init__(self, input_shape, dropout=None, n_blocks=4):
        super().__init__(input_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout
        tmin = self.dataset_config.tmin
        tmax = self.dataset_config.tmax
        frequency = self.config.transformations.eeg.new_freq
        dimensions = self.config.transformations.image.dimensions

        # Load ResNet model
        base_path = Path(__file__).resolve().parent
        base_model = models.resnet18()
        self.conv_layers = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        # Load TCNAE model to get avg pool size
        model_path = (base_path / f"../../trained_models/{TCNAE_EEG_DECODER_MODEL}.pth").resolve()
        tcnae_model = torch.load(model_path, map_location=self.device)

        self.avgpool_size = tcnae_model.avgpool_size
        self.latent_timepoints = int((((tmax - tmin) / 1000) * frequency) / self.avgpool_size)

        # Setup conv layers based on number of blocks
        basicblock_index = 4 + n_blocks
        self.conv_layers = self.conv_layers[0:basicblock_index] # 5 to 8

        # Set up avgpool layer with dynamic adpative avgpool size
        conv_output_dims = self.compute_output_shape(self.conv_layers, (3, dimensions, dimensions), flatten=False)
        adaptive_avgpool_size = int(round((512 / conv_output_dims[1]) ** 0.5)) # Match output size as close to original ResNet output size as possible ((512, 1, 1) = 512)
        self.avgpool = nn.AdaptiveAvgPool2d((adaptive_avgpool_size, adaptive_avgpool_size))
        
        # Set avgpool to match ResNet standard
        #self.avgpool = base_model.decoder.avgpool
        
        # Calculate ResNet output size after avgpool
        self.encoder_output = self.compute_output_shape(self.avgpool, conv_output_dims[1:])
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.encoder_output, out_features=256),
            nn.Dropout(self.dropout),    
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(self.dropout),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 4 * self.latent_timepoints)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = x.view(-1, 4, self.latent_timepoints) # first dimension is 4 because the TCNAE encoder compresses data into 4 channels
        return x
    
class TCNAEEncoder(Encoder):
    def __init__(self, input_shape, avgpool_size):
        super().__init__(input_shape)
        self.avgpool_size = self.config.models.tcnae.avgpool_size if avgpool_size is None else avgpool_size

        # Dilation numbers excl. initial (1)
        self.dilations = [2 ** i for i in range(1, 7)]

        self.init_dilated_conv = nn.Conv1d(input_shape[1], 64, kernel_size=8, padding='same')
        self.init_pointwise_conv = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)

        # Dilated convolutional layers
        self.dilated_convs = nn.ModuleList()
        for _, dilation in enumerate(self.dilations):
            self.dilated_convs.append(nn.Conv1d(in_channels=16, out_channels=64, kernel_size=8, dilation=dilation, padding='same'))
            
        # 1x1 Convolutions for skip connections
        self.pointwise_convs = nn.ModuleList([nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1) for _ in range(len(self.dilations))])
        
        # Final 1x1 Convolution
        self.final_conv = nn.Conv1d(in_channels=16 * 7, out_channels=4, kernel_size=1)
        self.avg_pool = nn.AvgPool1d(self.avgpool_size)
        
        combined_layers = [self.init_dilated_conv, self.init_pointwise_conv] + \
                  [layer for pair in zip(self.dilated_convs, self.pointwise_convs) for layer in pair] + \
                  [self.final_conv]
        #self.encoder_output = self.compute_output_shape([nn.Sequential(*combined_layers)], (3, 10))
        self.encoder_output = 4 * 32 * 32

    def forward(self, x):
        intermediate_outputs = []
        x = self.init_dilated_conv(x)
        x = F.relu(x)
        x = self.init_pointwise_conv(x)
        intermediate_outputs.append(x)
        
        for dilated_conv, pointwise_conv in zip(self.dilated_convs, self.pointwise_convs):
            x = dilated_conv(x)
            x = F.relu(x)
            x = pointwise_conv(x)
            intermediate_outputs.append(x)
        
        concatenated_output = torch.cat(intermediate_outputs, dim=1)
        x = self.final_conv(concatenated_output)
        x = self.avg_pool(x)
        return x