import numpy as np
import torch.nn as nn
import torch

from opus_binding import LibOPUSBinding
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

class YAAPTImplementationEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lib = LibOPUSBinding()
        self.output_size = 1
        self.dummy_parameter = nn.Parameter(torch.rand(1))

    
    @staticmethod
    def extract_pitch_single(sample, fs=16000):
        padded_sample = np.pad(sample, (0, 320), mode='constant')
        signal = basic.SignalObj(data=padded_sample, fs=fs)
        pitch = pYAAPT.yaapt(signal, frame_length=20, frame_space=10, f0_min=62.5, f0_max=500)
        return pitch.samp_values

    def forward(self, x: torch.Tensor):
        x_np = x.cpu().numpy()
        signal = basic.SignalObj(data=x_np,fs=16000)
        out = torch.as_tensor(np.asarray(list(map(lambda sample: self.extract_pitch_single(sample, fs=16000), x_np))),device=x.device)
        out = torch.clamp(torch.log2(out/62.5)*60, min=0, max=179.49).round().long()
        return out
    
class YAAPTImplementationDecoder(nn.Module):
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