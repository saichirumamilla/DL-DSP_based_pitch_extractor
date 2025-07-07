import numpy as np
import torch.nn as nn
import torch

from opus_binding import LibOPUSBinding
import torchcrepe
import amfm_decompy.basic_tools as basic

class CREPEImplementationEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lib = LibOPUSBinding()
        self.output_size = 1
        self.dummy_parameter = nn.Parameter(torch.rand(1))

    @staticmethod
    def extract_pitch_single(sample, fs=16000):
        batch_size = sample.shape[0]
        padded_sample = np.pad(sample, (0, 320), mode='constant')
        signal = basic.SignalObj(data=padded_sample, fs=fs)
        #pitch = pYAAPT.yaapt(signal, frame_length=20, frame_space=10, f0_min=62.5, f0_max=500)
        pitch = torchcrepe.predict(signal, fs, hop_length = 160, fmin=62.5, fmax = 500, model = 'tiny', batch_size=batch_size)
        return pitch

    def forward(self, x: torch.Tensor):
        #x_np = x.cpu().numpy()
        batch_size = x.shape[0]

        out = torchcrepe.predict(x, 16000, hop_length = int(16000/100.), fmin=62.5, fmax = 500, model = 'tiny', batch_size=batch_size)
        
        out = torch.clamp(torch.log2(out/62.5)*60, min=0, max=179.49).round().long()
        return out
    
class CREPEImplementationDecoder(nn.Module):
    output_size = 180
    min_freq = 62.5
    max_freq = 500
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.rand(1))
        idx_pitch = torch.exp2(torch.linspace(torch.log2(torch.tensor(self.min_freq)), torch.log2(torch.tensor(self.max_freq)), self.output_size))
        self.idx_pitch = torch.nn.Parameter(idx_pitch, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.one_hot(x.long(), num_classes=self.output_size).float()