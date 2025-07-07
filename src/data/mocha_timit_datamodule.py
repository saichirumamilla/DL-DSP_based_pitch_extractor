from typing import Any, Dict, Optional, Tuple, Callable
import os
import zipfile
import random
import urllib.request
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
import librosa
import json


import torch
import torchaudio
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import torch.utils.data.dataloader


from src.data.components.base import BaseWrapper
from src.models.components.decoders import Decoder

from src.utils import RankedLogger
from opus_binding import LibOPUSBinding

log = RankedLogger(__name__, rank_zero_only=True)

# librosa requires numba, disable its debug logger
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def collate_var_len_tensor_fn(batch, *, collate_fn_map=None):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

class MTIMITDataset(Dataset):
    eps = 1e-5
    def __init__(self, file_list, output_type, sample_rate, frame_rate, slice_length, slice,transforms):
        self.file_list = file_list
        self.lib = LibOPUSBinding()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.output_type = output_type
        self.slice_length = slice_length
        self.slice = slice
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)
    
    def load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    def get_gender(self, metadata_file):
     try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            gender = metadata.get('sex')
            if gender is None:
                raise KeyError("Gender information not found in metadata.")
            return gender
     except FileNotFoundError:
        print(f"Metadata file '{metadata_file}' not found.")
        return None
     except json.JSONDecodeError:
        print(f"Error decoding JSON in metadata file '{metadata_file}'.")
        return None
     
    def transform_pitch(self, pitch):
        if self.output_type == 'hz':
            mask = torch.logical_or(pitch < 62.5, pitch > 500)
            return pitch, mask
        elif self.output_type == 'class_label':
            mask = torch.logical_or(pitch < 62.5, pitch > 500)
            pitch = torch.clamp(torch.log2(pitch/62.5)*60, min=0, max=179.49).round().long()
            pitch_onehot = torch.nn.functional.one_hot(pitch, num_classes=180).float()
            i = torch.arange(-120, 121, 20)
            kernel = torch.exp(-i**2 / (1250))
            pitch_blurred = torch.nn.functional.conv1d(pitch_onehot[:, None], kernel[None, None, :], padding='same')
            return (pitch_blurred / pitch_blurred.sum(dim=-1, keepdim=True)).squeeze(1), mask
        

    def calculate_rms(self, speech):
        frame_length = 320 
        hop_length = 160 
        pad_length = int(frame_length // 2)
        if len(speech) % hop_length != 2*frame_length:
            speech = np.pad(speech, (0, pad_length), mode='constant')
        rmse = librosa.feature.rms(y=speech, frame_length=frame_length, hop_length=hop_length, center=False)[0]
        return torch.from_numpy(rmse)

    def __getitem__(self, idx):

        file_path = Path(self.file_list[idx])
        speech, sample_rate = torchaudio.load(file_path)
        speech = speech[0]
        if sample_rate != self.sample_rate:
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(speech)

        
        speech = self.transforms(speech)
        
        if self.slice:
            if len(speech) < self.slice_length * self.sample_rate:
                speech = F.pad(input = speech,pad = (0, self.slice_length * self.sample_rate - len(speech)),mode='constant',value=0)
            #randomly slice 1 second of audio if longer than 1 second                
            start_time = random.randint(0, len(speech) - self.slice_length * self.sample_rate)
            end_time = start_time + self.slice_length * self.sample_rate

        else:
            start_time = 0
            end_time = len(speech)
            self.slice_length = len(speech) 

        speech = speech[start_time:end_time]
        
        dset_root = file_path.parent.parent
        if dset_root.name == "TIMIT":
            consensus_data_folder = f"timit_consensus_{file_path.parent.name}"
        elif dset_root.name == "MOCHA_TIMIT":
            consensus_data_folder = f"mocha_timit_consensus_{file_path.parent.name}"
        elif dset_root.name == "KEELE":
            consensus_data_folder = f"keele_consensus_{file_path.parent.name}"
        elif dset_root.name == "FDA_sr20":
            consensus_data_folder = f"cstr_consensus_{file_path.parent.name}"
        elif dset_root.name == "TIMIT_training_data":
            consensus_data_folder = f"timit_consensus_{file_path.parent.name}"
        else:
            raise ValueError("No Consensus data found for the dataset.")
        pitch_file = dset_root / consensus_data_folder / "pitch.npy"
        pitch = torch.tensor(np.load(pitch_file))
        probability_file = dset_root / consensus_data_folder / "probability.npy"
        probability = torch.tensor(np.load(probability_file))
        metadata_file =  dset_root / consensus_data_folder / "_metadata.json"
        rmse = self.calculate_rms(speech)
        
        if len(pitch) < self.slice_length * self.frame_rate:
            probability = F.pad(probability, pad=(0, self.slice_length * self.frame_rate - len(probability)), mode='constant', value=0)
            pitch = F.pad(pitch, pad=(0, self.slice_length * self.frame_rate - len(pitch)),mode='constant', value=0)
        pitch, mask = self.transform_pitch(pitch[start_time * self.frame_rate // self.sample_rate :(end_time * self.frame_rate // self.sample_rate) ])
        probability = probability[start_time * self.frame_rate // self.sample_rate:end_time * self.frame_rate // self.sample_rate]
        

        mask = mask.bool() | (rmse < 0.025 * rmse.max()).bool() | (probability < 0.5).bool()
        probability[mask] = 0
        pitch[mask] = 0

        if dset_root.name == "TIMIT":
            gender = self.get_gender(metadata_file)
            return speech, pitch, probability, rmse, gender
        else:
            return speech, pitch, probability
        

class MTIMITDataModule(pl.LightningDataModule):
    def __init__(
        self,
        corpora_dir: str | None = None,
        data_dir: str = "data/",
        dataset: str = "TIMIT", 
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 32,
        num_workers: int = 8,
        target_freq: int = 16000,
        frame_rate: int = 100,
        slice_length: float = 1,
        slice =  True,
        output_type: str = "hz",
        features: dict[str, BaseWrapper] = {}, 
        persistent_workers: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        transforms: list[Callable] = [],
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.corpora_dir = corpora_dir
        self.data_dir = data_dir
        self.dataset = dataset
        self.target_freq = target_freq
        self.frame_rate = frame_rate
        self.output_type = output_type
        self.features = features
        self.train_val_test_split = train_val_test_split  
        self.drop_last = drop_last
        self.slice_length = slice_length
        self.slice = slice
        self.transforms = transforms

        # Assign collate function to instance variable
        collate_fn_map = deepcopy(torch.utils.data.dataloader._utils.collate.default_collate_fn_map)
        collate_fn_map[torch.Tensor] = collate_var_len_tensor_fn
        self.collate_fn = partial(torch.utils.data.dataloader._utils.collate.collate, collate_fn_map=collate_fn_map)

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        decoder = Decoder()
        return decoder.output_size

    def prepare_data(self) -> None:      # download and prepare data
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        print("Preparing data...")
        # determine the device to use
        # device = self.trainer.strategy.root_device
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
       
       # loader = DataLoader(speech_signals, batch_size=1,num_workers=8, collate_fn=features_dict, pin_memory=True, pin_memory_device=str(device))
        print("Data directory:", self.data_dir)
        zip_file_path = os.path.join(self.data_dir, f"{self.dataset}.zip")
        data_folder_path = os.path.join(self.data_dir, f"{self.dataset}")
        if self.dataset == "MOCHA_TIMIT":
            if not os.path.exists(data_folder_path):
            # Data is not present, download it
                if not os.path.exists(zip_file_path):
                    url = "https://zenodo.org/records/3920591/files/MOCHA_TIMIT.zip"  # URL FOR DATASET
                    print("Downloading data...")
                    urllib.request.urlretrieve(url, zip_file_path)
                    print("Data downloaded successfully.")
                print("Extracting data from ZIP file...")
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("Data extracted successfully.")
            else:
                print("Data folder already exists, skipping download and extraction.")

        else:
            print("Should implement the code to check if data is already downloaded and extracted for TIMIT")
        
        data_folder = os.path.join(self.data_dir, f"{self.dataset}")
        file_list = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file == "signal.wav"]
        mtimit_dataset = MTIMITDataset(file_list, self.output_type, frame_rate=self.frame_rate, slice_length=self.slice_length, sample_rate=self.target_freq,slice=self.slice,transforms=self.transforms)
        print("Data prepared successfully.")
        


    def setup(self, stage: Optional[str] = None) -> None:   # split data
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        data_folder = os.path.join(self.data_dir, f"{self.dataset}")
        print("Data folder:", data_folder)
        file_list = [os.path.join(root, file) for root, _, files in os.walk(data_folder) for file in files if file == "signal.wav"]
        if not file_list:
            raise FileNotFoundError("No wav files found in the dataset directory.")
        if not self.data_train:
        # Split the dataset into train, val, and test sets
            mtimit_dataset = MTIMITDataset(file_list, self.output_type, frame_rate=self.frame_rate, slice_length=self.slice_length, sample_rate=self.target_freq,slice=self.slice,transforms=self.transforms)
            train_dataset, val_dataset, _ = random_split(
                dataset=mtimit_dataset, lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42)  
            )
            mtimit_dataset_2 = MTIMITDataset(file_list, self.output_type, frame_rate=self.frame_rate, slice_length=self.slice_length, sample_rate=self.target_freq, slice=self.slice, transforms=self.transforms)
            _, val_dataset, test_dataset = random_split(
                dataset=mtimit_dataset, lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42)  
            )
        # Assign the split datasets to instance variables
            
            self.data_train = train_dataset
            self.data_val = val_dataset
            self.data_test = test_dataset
            print("batch_size", self.batch_size_per_device)
            print("Train set length:", len(self.data_train))
            print("Validation set length:", len(self.data_val))
            print("Test set length:", len(self.data_test))
        print("Data set-up done successfully.")

        

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        print("Test dataloader")    
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            persistent_workers=self.hparams.persistent_workers,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,      
            shuffle=False,
            drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """   
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            persistent_workers=self.hparams.persistent_workers,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=self.drop_last
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            persistent_workers=self.hparams.persistent_workers,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
            drop_last=self.drop_last
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MTIMITDataModule()
