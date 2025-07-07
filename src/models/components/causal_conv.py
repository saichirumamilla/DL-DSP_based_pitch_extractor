from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        # Padding only along the height and width dimensions
        padding_time = (self.kernel_size[0] // 2 * 2 * dilation, 0)
        padding_freq = (self.kernel_size[1] // 2 * dilation, self.kernel_size[1] // 2 * dilation)
        self.padding = torch.nn.ConstantPad2d(padding_freq + padding_time, 0)
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # Pad the input tensor only along the height and width dimensions
        padded_input = self.padding(input)
        # Perform the 2D convolution
        return F.conv2d(padded_input, weight, bias, self.stride, 0, self.dilation, self.groups)
