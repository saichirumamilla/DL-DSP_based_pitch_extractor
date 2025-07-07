import numpy as np
import scipy
import scipy.signal 
import librosa
import matplotlib.pyplot as plt


import torch
import torchaudio
import torchaudio.functional
from torch import nn
from tensordict import TensorDict
import torch.nn.functional as F
from LPCTorch.lpctorch.lpc import LPCCoefficients

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
            self.lpc_res = LPCResidual()
            self.lpcr = False # make it true if you want to compute lpc residual
  
    def extract(self, x):
        '''
        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, num_channels)

        Returns:
            xcorr (torch.Tensor): Cross-correlation tensor of shape (batch_size, num_frames, TAU)
        '''
        batch_size, num_frames, frame_length = x.shape
        xcorr = torch.zeros(batch_size, num_frames, self.TAU, device=x.device)
        for tau in range(self.TAU):
            # obtain delayed frames by lag t
            x_tau = x.reshape(batch_size, -1).roll(-tau, dims=1).reshape(batch_size, num_frames, frame_length)
            # compute cross-correlation
            xcorr_tau = torch.sum(x * x_tau, dim=-1)
            xcorr_tau = 2 * xcorr_tau / (torch.sum(x * x, dim=-1) + torch.sum(x_tau * x_tau, dim=-1) + self.eps) 
            
            xcorr[:, :, tau] = xcorr_tau
        xcorr = xcorr[:,:,33:]
        return xcorr

    def prepare_input(self, x):
        '''
        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)
        
        Returns:
            LPC residual of shape (batch_size, num_frames, num_channels) if LPC should be computed else returns the framed signal
        '''
        if self.lpcr == True:
            lpc_res=self.lpc_res(x) # taking x as([batch_size,no_of samples]) and return the lpc residual as [batch_size,nu_of frames,no_of elements]
            return lpc_res
        else:
            batch_size, num_samples = x.shape
            padding = torch.zeros(batch_size, self.H, device=x.device)
            x_padded = torch.cat((x, padding), dim=1)
            num_samples_padded = num_samples + self.H
            num_frames = (num_samples_padded - self.N) // self.H + 1
            frames = x_padded.unfold(1, self.N, self.H)
            return frames
    
class LPCResidual(FeatureExtractor):
    """LPC Residual

    The LPCResidual uses the output of the LPCCoefficients to compute the
    residual of the original signal.
    """
    def __init__(self, N=320, H=160, TAU=257):
        super().__init__(N=N, H=H)
        self.N = N
        self.H = H
        self.lpc = LPCCoefficients(sr=16000, duration=320/16000, order=20, padded=True)

    def synthesis(self, residual: torch.Tensor, lpcc: torch.Tensor) -> torch.Tensor:
        '''
            residual (torch.Tensor): Residual tensor of shape (batch_size, num_frames, num_elements_in_frame)
            lpcc (torch.Tensor): LPCC tensor of shape (batch_size, num_frames, num_elements_in_frame) 
        '''
        a = torch.ones_like(lpcc)
        reconstructed = torchaudio.functional.lfilter(residual.reshape(-1, self.N), 
                                                      lpcc.reshape(-1,self.lpc.p), 
                                                      a.reshape(-1,self.lpc.p))
        return reconstructed

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Parameters
        ----------
        X: torch.Tensor
           Input signal to be processed.
           Expected input is [Batch, Samples]

        Returns
        -------
        residual: torch.Tensor
           Residual computed from input signal after LPC.
           Expected output is [Batch, Frames, Samples]
        """
        lpcc: torch.Tensor = self.lpc(x)
        a = torch.ones_like(lpcc)
        x_frames = self.lpc.frames(x)
        residual = torchaudio.functional.lfilter(x_frames.reshape(-1, self.N), 
                                                 a.reshape(-1,self.lpc.p), 
                                                 lpcc.reshape(-1,self.lpc.p))
        residual = residual.reshape_as(x_frames)
        return residual
        
    
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

    def extract(self, x):
        '''
        Arguments:
            x (torch.Tensor): Complex-valued STFT of shape (batch_size, num_frames, num_freqs)

        Returns:
            x_if (torch.Tensor): IF features of shape (batch_size, num_frames, 3*num_freqs)
        '''
        x_logmspec = torch.clamp(x.abs(), 1e-6).log()
        delta_x = x * x.roll(1, -2).conj()
        delta_x = delta_x / (delta_x.abs() + self.eps)
        # first element will be wrong
        x_logmspec = x_logmspec[:, 1:]
        delta_x = delta_x[:, 1:]
        x_if = torch.concat([x_logmspec, delta_x.real, delta_x.imag], dim=-1)
        return x_if

    def prepare_input(self, x):
        '''
        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples)

        Returns:
            x_stft (torch.Tensor): Complex-valued STFT tensor of shape 
                (batch_size, num_frames, num_freqs)
        '''
        x_stft: torch.Tensor = self.stft(x)[..., :self.N_freq, :].transpose(-1, -2)
        return x_stft


class ParallelFeatureExtractor(nn.ModuleDict):
    """Runs modules in parallel on the same input and merges their results."""

    def __init__(self, **modules: nn.Module):
        super().__init__(modules)

    def forward(self, x: torch.Tensor) -> TensorDict:
        outputs = {}
        for key, extractor in self.items():
            outputs[key] = extractor(x)  # Applying the current encoder
        
        return TensorDict(outputs, batch_size=x.shape[0])
