from typing import Callable
from tensordict import TensorDict

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size: int, min_freq: float, max_freq: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        idx_pitch = torch.exp2(torch.linspace(torch.log2(torch.tensor(min_freq)), torch.log2(torch.tensor(max_freq)), output_size))
        self.idx_pitch = torch.nn.Parameter(idx_pitch, requires_grad=False)

    def forward(self, x: TensorDict):
        raise NotImplementedError("This is a base class. Please override this method in a subclass.")
    
class GRUDecoder(Decoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation: Callable[..., nn.Module],
                 min_freq: float,
                 max_freq: float,
                 ):
        super().__init__(input_size, hidden_size, output_size, min_freq, max_freq)

        self.blocks = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            activation(),
            nn.Linear(self.hidden_size, self.hidden_size),
            activation(),
        )
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.blocks_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x: TensorDict):
        x = self.blocks(x)
        x, _ = self.gru(x)
        x = self.blocks_2(x)
        return x