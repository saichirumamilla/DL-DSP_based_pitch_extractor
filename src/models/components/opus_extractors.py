import numpy as np
import scipy
import scipy.signal 
import librosa
import matplotlib.pyplot as plt
import sys
from pathlib import Path


import torch
import torchaudio
import torchaudio.functional
from torch import nn
from tensordict import TensorDict
import torch.nn.functional as F

from opus_binding import LibOPUSBinding

class FeatureExtractor(nn.Module):
    eps = 1e-5
    def __init__(self, N=None, H=None):
        super().__init__()
        self.N = N
        self.H = H

    def extract(self, x: torch.Tensor):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, num_elements_in_frame)
        '''
        raise NotImplementedError("This is a base class. Please override this method in a subclass.")
    
    def prepare_input(self, x: torch.Tensor):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)
        '''
        raise NotImplementedError("This is a base class. Please override this method in a subclass.")

    def forward(self, x: torch.Tensor):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)
        '''
        
        x = self.prepare_input(x)
        
        return self.extract(x)
    
   
class XCorrExt(FeatureExtractor):
    def __init__(self, N=320, H=160, TAU=257):
            super().__init__(N=N, H=H)
            self.TAU = TAU
            self.N= N
            self.lib = LibOPUSBinding()
  
    def extract(self, x):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, num_elements_in_frame)
        '''
        xcorr = torch.as_tensor(np.asarray(list(map(lambda xi: self.lib.lossless_encode_array(xi)[0], x.cpu()))), dtype=x.dtype, device=x.device)
        return xcorr

    def prepare_input(self, x):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)
        '''
        
        return x
    

class IFExt(FeatureExtractor):
    def __init__(self, N=320, H=160, N_freq=30):
            super().__init__(N=N, H=H)
            self.N_freq = N_freq
            self.H=H
            self.stft = torchaudio.transforms.Spectrogram(n_fft=N, 
                                                          hop_length=H, 
                                                          power=None, 
                                                          center=True, 
                                                          pad_mode="constant")
            self.lib = LibOPUSBinding()

    

    def extract(self, x):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, num_elements_in_frame)
        '''
        x_if = torch.as_tensor(np.asarray(list(map(lambda xi: self.lib.lossless_encode_array(xi)[1], x.cpu()))), device=x.device)
        x_if= x_if.to(torch.device('cuda'))
        return x_if.float().transpose(1,2)

    def prepare_input(self, x):
        '''
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)
        '''
        
        return x


class ParallelFeatureExtractor(nn.ModuleDict):
    """Runs modules in parallel on the same input and merges their results."""

    def __init__(self, **modules: nn.Module):
        super().__init__(modules)

    def forward(self, x: torch.Tensor) -> TensorDict:
        outputs = {}
        for key, extractor in self.items():
            outputs[key] = extractor(x)  # Applying the current encoder
        
        return TensorDict(outputs, batch_size=x.shape[0])
