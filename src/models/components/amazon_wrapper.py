import numpy as np
import torch.nn as nn
import torch

from opus_binding import LibOPUSBinding

class AmazonImplementationEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lib = LibOPUSBinding()
        self.output_size = 1
        self.dummy_parameter = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor):
        x_np = x.cpu().numpy()
        out = torch.as_tensor(np.asarray(list(map(self.lib.extract_pitch, x_np))), device=x.device)
        out = (60*(out[..., 0] + 1.5)).round()
        return out
    
class AmazonImplementationDecoder(nn.Module):
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