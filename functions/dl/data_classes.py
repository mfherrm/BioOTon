from itertools import compress
import os
import pandas as pd

import random
from sklearn.preprocessing import LabelEncoder

import torch
import torchaudio as ta
from torchcodec.decoders import AudioDecoder
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader




from functions.dl.convenience_functions import to_device

class SpectroDataset(Dataset):
    def __init__(self, 
                recording_path: list[str], 
                label_path: list[str],
                #  transform: None | Callable,
                sampling_rate: int = 44100, #Hz
                loudness: int = 10,
                device = 'cpu',
                **kwargs):
        # Assign instance variables
        self.l_path = label_path
        self.r_path = recording_path
        self.device = device

        # Load point labels
        dir_files = os.listdir(self.l_path)
        soundscape_file = next(f for f in dir_files if f.endswith("_single.parquet"))
        sound_df = pd.read_parquet(os.path.join(self.l_path, soundscape_file))
        sound_df = sound_df.drop(sound_df.loc[sound_df.label.isna()].index)

        # Load processed sound waves
        rec_ids = {int(f.split('_')[0]) for f in os.listdir(self.r_path) if f.endswith(".pt")}

        # Find common subset
        self.file_df = sound_df[sound_df['id'].isin(rec_ids)].dropna(subset=['label'])

        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.file_df['label'])

        self.fileNames = self.file_df.id.values

        # for testing sample only seconds 5 to 10
        self.llimit = int(16000 * 5)
        self.rlimit = int(self.llimit * 2)
    
    def __len__(self):
        return len(self.fileNames)
    
    def __getitem__(self, idx:int):
        #x = torch.zeros(10*self.sr)
        print("Getting", os.path.join(self.r_path, f"{self.fileNames[idx]}_audio.pt"))

        wave = torch.load(os.path.join(self.r_path, f"{self.fileNames[idx]}_audio.pt"))

        # decoder = AudioDecoder(os.path.join(self.r_path, f"{self.fileNames[idx]}_audio.flac"))
        # result = decoder.get_all_samples()

        # wave = result.data.to(torch.float32)#.unsqueeze(0)
        # sr = result.sample_rate

        # # Move to specified device, e.g. GPU
        # # wave = wave.to(torch.float32)
        
        # # Resample to the sampling rate given by the args 
        # if sr != self.sampling_rate:
        #     print(f"Initial sampling rate is {sr}, resampling to {self.sampling_rate}")
        #     wave = ta.functional.resample(wave, sr, self.sampling_rate)

        # # LUFS normalization to a given loudness
        # wave_loudness = AF.loudness(wave, self.sampling_rate)
        # gain = self.loudness - wave_loudness
        # multiplier = 10 ** (gain/20)
        # wave = wave * multiplier
        
        wave = wave[self.llimit:self.rlimit]
        # wave = wave[:, int(llimit):int(llimit*2)]

        # most of the time the first and last 10 seconds should be cut off 
        # wave = wave[:, int(llimit):int(llimit*5)+1]

        # Squeeze to go from 1,x to x shape tensor
        # wave = wave.squeeze()
        # wave = to_device(wave.squeeze(), self.device)
        # return wave, sr, int(self.fileLabels[idx])
        return wave, int(self.encoded_labels[idx])
    
class SpectroDataLoader(DataLoader):
    """
    """

    def __init__(self, datas, batch_size, samples: list[int], device='cpu'):
        self.waves = []
        self.lbs = []
        self.datas = datas
        self.batch_size = batch_size
        # self.sr = []
        self.device = device
        self.samples = samples
        self.n_samples = len(self.samples)

    def __iter__(self):
        # Shuffle all waves
        random.shuffle(self.samples)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.samples[i : i + self.batch_size]
            
            wvs = []
            las = []

            for idx in batch_indices:
                wave, label = self.datas[idx]
                wvs.append(wave)
                las.append(label)

            yield to_device([torch.stack(wvs), torch.Tensor(las)], self.device)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size