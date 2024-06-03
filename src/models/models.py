#__all__ = ['SimpleEEGReconstruction', 'AlexNet']

from utils.config_loader import ConfigLoader
from models.encoders import BaselineEncoder, AlexNetEncoder, ResNetEncoder, TCNAEEncoder, PretrainedResNetLatentEncoder, RWResNetLatentEncoder, SqueezeNetEncoder
from models.decoders import BaselineDecoder, AlexNetEEGDecoder, AlexNetClassDecoder, ResNetEEGDecoder, ResNetClassDecoder, ResNetLatentEEGDecoder, CustomEEGDecoder, TCNAEDecoder, SqueezeNetClassDecoder
import torch


# -------- Base Model class -------- #

class Model(torch.nn.Module):
    def __init__(self, input_shape, output_shape, dropout=None):
        self.dropout = dropout
        super().__init__()
        self.config = ConfigLoader()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.set_encoder()
        self.set_decoder()
        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

    def set_encoder(self):
        raise NotImplementedError

    def set_decoder(self):
        raise NotImplementedError

    def forward(self, input):
        """
        Forward pass through network. Simply calls encoder and decoder forward methods.
        
        Parameters
        ----------
            input : torch.Tensor
                Input data.
        
        Returns
        -------
            torch.Tensor
                Output of network.
        """
        return self.decoder(self.encoder(input))

# -------- Specific Models -------- #

class BaselineEEGReconstruction(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        self.encoder = BaselineEncoder(self.input_shape, self.dropout)

    def set_decoder(self):
        self.decoder = BaselineDecoder(self.encoder.encoder_output, self.output_shape)
    
    def forward(self, input):
        return super().forward(input)

class AlexNet(Model):
    def __init__(self, input_shape, output_shape, dropout=None):
        self.dropout = dropout
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        print("Input", self.input_shape)
        self.encoder = AlexNetEncoder(self.input_shape)

    def set_decoder(self):
        print("Output", self.output_shape)
        # Choose decoder to use based on the specified task we want to solve
        if self.config.task == 'eeg':
            self.decoder = AlexNetEEGDecoder(self.encoder.encoder_output, self.output_shape, self.dropout)
        elif self.config.task == 'class':
            self.decoder = AlexNetClassDecoder(self.encoder.encoder_output, self.output_shape, self.dropout)
    
    def forward(self, input):
        return super().forward(input)

class SqueezeNet(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        print("Input", self.input_shape)
        self.encoder = SqueezeNetEncoder(self.input_shape)

    def set_decoder(self):
        print("Output", self.output_shape)
        # Choose decoder to use based on the specified task we want to solve
        if self.config.task == 'eeg':
            raise NotImplementedError
        elif self.config.task == 'class':
            self.decoder = SqueezeNetClassDecoder(self.encoder.encoder_output, self.output_shape)
    
    def forward(self, input):
        return super().forward(input)
    
class ResNet(Model):
    def __init__(self, input_shape, output_shape, dropout=None, n_blocks=4):
        self.dropout = dropout
        self.n_blocks = n_blocks
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        print("Input", self.input_shape)
        self.encoder = ResNetEncoder(self.input_shape, self.n_blocks)

    def set_decoder(self):
        print("Output", self.output_shape)
        # Choose decoder to use based on the specified task we want to solve
        if self.config.task == 'eeg':
            self.decoder = ResNetEEGDecoder(self.encoder.encoder_output, self.output_shape, self.dropout)
        elif self.config.task == 'class':
            self.decoder = ResNetClassDecoder(self.encoder.encoder_output, self.output_shape)
    
    def forward(self, input):
        return super().forward(input)
    
class TCNAE(Model):
    def __init__(self, input_shape, output_shape, avgpool_size=4):
        self.avgpool_size = avgpool_size
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        print("Input", self.input_shape)
        self.encoder = TCNAEEncoder(self.input_shape, self.avgpool_size)

    def set_decoder(self):
        print("Output", self.output_shape)
        self.decoder = TCNAEDecoder(self.encoder.encoder_output, self.output_shape, self.avgpool_size)
    
    def forward(self, input):
        return super().forward(input)

class ResNetLatentEEG(Model):
    def __init__(self, input_shape, output_shape, dropout=None, n_blocks=4):
        self.dropout = dropout
        self.n_blocks = n_blocks
        super().__init__(input_shape, output_shape)

    def set_encoder(self):
        print("Input", self.input_shape)
        self.encoder = PretrainedResNetLatentEncoder(self.input_shape, self.dropout, self.n_blocks)

    def set_decoder(self):
        print("Output", self.output_shape)
        self.decoder = ResNetLatentEEGDecoder(self.encoder.encoder_output, self.output_shape)
    
    def forward(self, input):
        return super().forward(input)