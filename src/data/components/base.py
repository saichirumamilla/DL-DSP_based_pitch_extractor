import torch
import torch.nn as nn

class BaseWrapper(nn.Module):
    def __init__(self, frame_rate: int, is_target: bool = False):
        super().__init__()
        self.frame_rate = nn.Parameter(torch.tensor(frame_rate), requires_grad=False)
        self.is_target = nn.Parameter(torch.tensor(is_target, dtype=torch.bool), requires_grad=False)
        self.force_cpu = False

    def forward(self, x):
        raise NotImplementedError('forward method must be implemented in the derived class.')
