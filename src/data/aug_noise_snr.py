import os
import numpy as np
import torchaudio
import torch
from random import choice
import math

def write_wav(file_path, sample_rate, data):
    torchaudio.save(file_path, torch.from_numpy(data).unsqueeze(0), sample_rate)

def snr_to_noise_scale(snr_db, signal_power, noise_power):
  
    snr_linear = 10 ** (snr_db / 10)
    return np.sqrt(signal_power / (snr_linear * noise_power))

def augment_with_noise(timit_file, noise_files, output_file, snr_db):
    timit_data, timit_sr = torchaudio.load(timit_file)
    noise_file = choice(noise_files)
    noise_data, noise_sr = torchaudio.load(noise_file)

    if noise_sr != timit_sr:
        raise ValueError(f"Sample rates do not match: {noise_sr} != {timit_sr}")

    noise_length = noise_data.shape[1]
    timit_length = timit_data.shape[1]

    if noise_length < timit_length:
        repeat_count = math.ceil(timit_length / noise_length)
        noise_data = noise_data.repeat(1, repeat_count)
    
    noise_data = noise_data[:, :timit_length]  

    timit_data_np = timit_data.numpy().flatten()
    noise_data_np = noise_data.numpy().flatten()

    # Calculate signal and noise power
    signal_power = np.mean(timit_data_np ** 2)
    noise_power = np.mean(noise_data_np ** 2)

    # Calculate the scale factor for noise based on desired SNR
    noise_scale = snr_to_noise_scale(snr_db, signal_power, noise_power)

    # Scale the noise and add it to the signal
    augmented_data = timit_data_np + (noise_data_np * noise_scale)

    # Clip the augmented data to the range [-1, 1]
    augmented_data = np.clip(augmented_data, -1.0, 1.0)

    write_wav(output_file, timit_sr, augmented_data)

def process_directory(timit_dir, noise_dir, output_dir, snr_db):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    noise_files = [os.path.join(root, f) for root, _, files in os.walk(noise_dir) for f in files if f.endswith('.wav')]

    for root, dirs, files in os.walk(timit_dir):
        for file in files:
            if file.endswith('.wav'):
                timit_file = os.path.join(root, file)

                relative_path = os.path.relpath(timit_file, timit_dir)
                output_file = os.path.join(output_dir, relative_path)

                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))

                augment_with_noise(timit_file, noise_files, output_file, snr_db)
                print(f'Processed {timit_file} -> {output_file} with SNR {snr_db} dB')

# Main script
timit_dir = 'data/TIMIT'
noise_dir = '/data/chsaikeerthi/demand'
output_base_dir = '/data/chsaikeerthi'

# Process with different SNR levels
for snr_db in [10,-10]:
    output_dir = os.path.join(output_base_dir, f'TIMIT_aug_noise_with_snr_{snr_db}')
    process_directory(timit_dir, noise_dir, output_dir, snr_db)
    print(f'Processed all files with SNR {snr_db} dB')
