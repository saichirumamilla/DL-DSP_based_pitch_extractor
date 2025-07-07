import os
import numpy as np
import torchaudio
import torch
from random import choice
import math

def write_wav(file_path, sample_rate, data):
    
    torchaudio.save(file_path, torch.from_numpy(data).unsqueeze(0), sample_rate)

def augment_with_noise(timit_file, noise_files, output_file):
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


    augmented_data = timit_data_np + noise_data_np

    
    augmented_data = np.clip(augmented_data, -1.0, 1.0) 

    write_wav(output_file, timit_sr, augmented_data)

def process_directory(timit_dir, noise_dir, output_dir):
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

                augment_with_noise(timit_file, noise_files, output_file)
                print(f'Processed {timit_file} -> {output_file}')


timit_dir = 'data/TIMIT'
noise_dir = '/data/chsaikeerthi/demand'
output_dir = '/data/chsaikeerthi/TIMIT_aug_noise'


process_directory(timit_dir, noise_dir, output_dir)
