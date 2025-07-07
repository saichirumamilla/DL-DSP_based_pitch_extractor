import os
import torchaudio
from torch.utils.data import Dataset
from src.models.components.extractors import FeatureExtractor
from opus_binding import LibOPUSBinding

class PitchDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.extractor = FeatureExtractor()
        self.files = os.listdir(folder)
        self.opus_binding = LibOPUSBinding()

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        file = self.files[idx]
        waveform, sample_rate = torchaudio.load(os.path.join(self.folder, file))
        features = self.extractor(waveform)
        pitch_values = self.opus_binding.extract_pitch(waveform.numpy())
        label = file.split('_')[0]
        return features, label , pitch_values