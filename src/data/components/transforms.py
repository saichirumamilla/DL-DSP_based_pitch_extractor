import torch
import numpy as np
import torchaudio
import os
from random import choice

def snr_to_noise_scale(snr_db, signal_power, noise_power):
    ''' 
    snr_db (float): Signal-to-noise ratio in dB
    Inputs the power of the signal, noise, and the SNR in dB 
    returns the scale factor to apply to the noise to achieve the desired SNR.
    '''
    snr_linear = 10 ** (snr_db / 10)
    return np.sqrt(signal_power / (snr_linear * noise_power))

def add_noise(x: torch.Tensor, snr_db: float):
    noise = torch.randn_like(x)
    signal_power = torch.mean(x ** 2).item()
    noise_power = torch.mean(noise ** 2).item()
    noise_scale = snr_to_noise_scale(snr_db, signal_power, noise_power)
    return x + noise * noise_scale

def load_noise_file(noise_dir: str):
    
    subdirectories = [os.path.join(noise_dir, subdir) for subdir in os.listdir(noise_dir) 
                      if os.path.isdir(os.path.join(noise_dir, subdir))]
    
    if not subdirectories:
        raise ValueError(f"No subdirectories found in directory: {noise_dir}")
    chosen_subdir = choice(subdirectories)
    noise_files = [os.path.join(chosen_subdir, file) for file in os.listdir(chosen_subdir) 
                   if file.endswith('.wav')]
    
    if not noise_files:
        raise ValueError(f"No noise files found in subdirectory: {chosen_subdir}")
    
    return choice(noise_files)

def aug_noise(x: torch.Tensor, snr_db: int, apply_noise: bool, noise_dir: str):
    if apply_noise:
        noise_file = load_noise_file(noise_dir)
        noise_data, _ = torchaudio.load(noise_file)
        noise_data = noise_data[0]  # Assuming the first channel
        
        # Ensure noise data is long enough
        if noise_data.size(0) < x.size(0):
            noise_data = torch.cat([noise_data] * (x.size(0) // noise_data.size(0) + 1), dim=0)
        noise_data = noise_data[:x.size(0)]
        
        x = add_noise(x, snr_db)
    
    return x
