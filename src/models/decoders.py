from utils.config_loader import ConfigLoader
from utils.constants import TCNAE_EEG_DECODER_MODEL
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch 
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet152_Weights, AlexNet_Weights, SqueezeNet1_0_Weights

# -------- Base Decoder class -------- #

class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_shape):
        super().__init__()
        self.config = ConfigLoader()
        self.input_size = input_size
        self.output_shape = output_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        raise NotImplementedError
    
# Activation functions as torch.nn module
class Sin(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)
    
# -------- Specific Decoders -------- #    

class BaselineDecoder(Decoder):
    def __init__(self, input_size, output_shape):
        super().__init__(input_size, output_shape)
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size, output_shape[1] * output_shape[2])
        )
    
    def forward(self, x):
        x = self.mlp_layers(x)
        x = x.view(-1, self.output_shape[1], self.output_shape[2])
        return x

class AlexNetEEGDecoder(Decoder):
    def __init__(self, input_size, output_shape, dropout=None):
        super().__init__(input_size, output_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout
        self.avgpool = nn.AdaptiveMaxPool2d((6, 6))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(input_size, 1000),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1000, output_shape[1] * output_shape[2])
        )
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.output_shape[1], self.output_shape[2])
        return x
    
class AlexNetClassDecoder(Decoder):
    def __init__(self, input_size, output_shape, dropout=None):
        super().__init__(input_size, output_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout

        base_model = models.alexnet(weights=AlexNet_Weights.DEFAULT if self.config.models.general.pretrained else None)
        base_model.classifier[0] = nn.Dropout(self.dropout)
        base_model.classifier[3] = nn.Dropout(self.dropout)
        base_model.classifier[6] = nn.Linear(in_features=base_model.classifier[6].in_features, out_features=output_shape[1]) # Change final FC layer to match desired output (number of classes to predict from)

        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class SqueezeNetClassDecoder(Decoder):
    def __init__(self, input_size, output_shape, dropout=None):
        super().__init__(input_size, output_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout

        base_model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT if self.config.models.general.pretrained else None)
        base_model.classifier[0] = nn.Dropout(self.dropout)
        base_model.classifier[1] = nn.Conv2d(512, self.output_shape[1], kernel_size=1)

        self.classifier = base_model.classifier

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
    
class ResNetEEGDecoder(Decoder):
    def __init__(self, input_size, output_shape, dropout=None):
        super().__init__(input_size, output_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout

        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT if self.config.models.general.pretrained else None)
        base_model.fc = nn.Linear(in_features=input_size, out_features=output_shape[1] * output_shape[2])

        self.fc_layers = nn.Sequential(
            base_model.fc
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = x.view(-1, self.output_shape[1], self.output_shape[2])
        return x
    
class ResNetClassDecoder(Decoder):
    def __init__(self, input_size, output_shape):
        super().__init__(input_size, output_shape)

        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT if self.config.models.general.pretrained else None)
        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=output_shape[1])

        self.avgpool = base_model.avgpool
        self.fc_layers = base_model.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
    
class ResNetLatentEEGDecoder(Decoder):
    def __init__(self, input_size, output_shape):
        super().__init__(input_size, output_shape)
        base_path = Path(__file__).resolve().parent

        model_path = (base_path / f"../../trained_models/{TCNAE_EEG_DECODER_MODEL}.pth").resolve()
        tcnae_model = torch.load(model_path, map_location=self.device)
        self.tcnaedecoder = torch.nn.Sequential(tcnae_model.decoder)

        # Freeze the layers
        for param in self.tcnaedecoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.tcnaedecoder(x)
        return x
    
class CustomEEGDecoder(Decoder):
    def __init__(self, input_size, output_shape, dropout=None):
        super().__init__(input_size, output_shape)
        self.dropout = self.config.models.general.dropout if dropout is None else dropout
        self.pretrained_classifier_outputs = 512

        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT if self.config.models.general.pretrained else None)
        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=self.pretrained_classifier_outputs)

        self.avgpool = base_model.avgpool
        self.fc_layers = nn.Sequential(
            base_model.fc,
            nn.SiLU(),
            nn.Linear(self.pretrained_classifier_outputs, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, output_shape[1] * output_shape[2])
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = x.view(-1, self.output_shape[1], self.output_shape[2])
        return x

class TCNAEDecoder(Decoder):
    def __init__(self, input_size, output_shape, avgpool_size):
        super().__init__(input_size, output_shape)
        self.avgpool_size = self.config.models.tcnae.avgpool_size if avgpool_size is None else avgpool_size

        # Reverse dilation numbers excl. initial (64)
        self.dilations = [2 ** i for i in range(6, 0, -1)]

        # Initial upsampling layer and conv layers (dilated + pointwise)
        self.upsample_layer = nn.Upsample(scale_factor=self.avgpool_size)
        self.init_dilated_conv = nn.Conv1d(in_channels=4, out_channels=64, dilation=64, kernel_size=8, padding='same')
        self.init_pointwise_conv = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1)

        # Dilated convolutional layers
        self.dilated_convs = nn.ModuleList()
        for _, dilation in enumerate(self.dilations):
            self.dilated_convs.append(nn.Conv1d(in_channels=16, out_channels=64, kernel_size=8, dilation=dilation, padding='same'))
            
        # 1x1 Convolutions for skip connections
        self.pointwise_convs = nn.ModuleList([nn.Conv1d(in_channels=64, out_channels=16, kernel_size=1) for _ in range(len(self.dilations))])
        
        # Final 1x1 Convolution
        self.final_conv = nn.Conv1d(in_channels=16 * 7, out_channels=output_shape[1], kernel_size=1)

    def forward(self, x):
        intermediate_outputs = []

        x = self.upsample_layer(x)
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
        return x