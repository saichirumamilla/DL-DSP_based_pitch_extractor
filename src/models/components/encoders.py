from typing import Callable
from tensordict import TensorDict

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F

from src.models.components.causal_conv import CausalConv2d


class Encoder(nn.Module):
    input_size: int
    output_size: int

    def forward(self, x: TensorDict):
        raise NotImplementedError("This is a base class. Please override this method in a subclass.")
    
class IFEncoder(Encoder):
    def __init__(self,
            input_size: int,
            output_size: int,
            activation: Callable[..., nn.Module] = nn.Tanh,
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lin1 = nn.Linear(self.input_size, self.output_size)
        self.act1 = activation() #

    def forward(self, x: TensorDict) -> torch.Tensor:
        input_tensor = x['IFExt']
        out = self.lin1(input_tensor)
        out = self.act1(out)
        return out


class XCorrEncoder(Encoder):
    def __init__(
            self, 
            input_size: int,
            num_channels: int,
            activation: Callable[..., nn.Module],
        ):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.conv1 = CausalConv2d(1,
                     num_channels,
                     kernel_size=(3, 3)
                     )
        self.act1 = activation()
        self.conv2 = CausalConv2d(
                     num_channels,
                     num_channels,
                     kernel_size=(3, 3)
                     )
        self.act2 = activation()
        self.conv3 = CausalConv2d(
                     num_channels,
                     1,
                     kernel_size=(3, 3)
                     )
        self.act3 = activation()
        #self.padding = (0, 0, 0, 0)  # (left, right, top, bottom)

    def forward(self, x: TensorDict) -> torch.Tensor:      
        input_tensor = x['XCorrExt'][:, None]
        out = self.conv1(input_tensor)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        return out[...,0,:,:]
    

class JointEncoder(nn.ModuleDict, Encoder):
    """Runs modules in parallel on the same input and merges their results."""
    def __init__(
            self,
            output_size: int,
            activation: Callable[..., nn.Module],
            **modules: nn.Module,
            ):
        modules = {key: value for key, value in modules.items() if isinstance(value, nn.Module)}
        super().__init__(modules)
        self.encoders = [key for key in modules.keys()]
        self.input_size = 0
        self.output_size = sum([m.output_size for m in self.values()])

    def forward(self, x: TensorDict) -> torch.Tensor:

        outputs = []
        for key in self.encoders:
            outputs.append(self[key](x))
            out = torch.cat(outputs, dim=-1)
        return out 

if __name__ == "__main__":
    # Test code
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    waveform, sample_rate = torchaudio.load('data/MOCHA_TIMIT/faet0_001/signal.wav')
    ifencoder = IFEncoder(input_size=90, output_size=64)
    xencoder = XCorrEncoder(input_size=224,num_channels=8,activation=nn.Tanh)
    encoder1 = JointEncoder(output_size=64,activation=nn.Tanh)
    print(xencoder)