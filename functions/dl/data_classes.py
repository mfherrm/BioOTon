from itertools import compress
import os
import pandas as pd

import random
from sklearn.preprocessing import LabelEncoder

import torch
import torchaudio as ta
import torchaudio.functional as AF
from torch.utils.data import Dataset



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
        self.l_path = label_path
        self.r_path = recording_path

        dir_files = os.listdir(label_path)
        soundscape_file = os.path.join(self.l_path, list(compress(dir_files, [file.endswith("_single.parquet") for file in dir_files]))[0])
        sound_df = pd.read_parquet(soundscape_file)
        sound_df = sound_df.drop(sound_df.loc[sound_df.label.isna()].index)

        dir_files = os.listdir(recording_path)
        dir_files = list(compress(dir_files, [file.endswith(".flac") for file in dir_files]))

        dir_files = [int(f.split('_')[0]) for f in dir_files]

        recording_df = pd.DataFrame({"id":dir_files})

        self.file_df = sound_df.merge(recording_df)
        
        self.fileNames = self.file_df.id.values
        self.fileLabels = self.file_df.label.values
        # self.transforms = transform
        self.sampling_rate = sampling_rate
        self.loudness = loudness
        self.device = device
    
    def __len__(self):
        return len(self.fileNames)
    
    def __getitem__(self, idx):
        #x = torch.zeros(10*self.sr)
        print("Getting", os.path.join(self.r_path, (str(self.fileNames[idx])+"_audio.flac")))
        # Uses native sampling rate of the file
        # Normalize arg does bit depth normalization
        # Move to device specified
        wave, sr = ta.load(uri=os.path.join(self.r_path, (str(self.fileNames[idx])+"_audio.flac")))#, normalize=True)
        
        # Move to specified device, e.g. GPU
        wave = wave.to(torch.float32)
        
        # Resample to the sampling rate given by the args 
        if sr != self.sampling_rate:
            print(f"Initial sampling rate is {sr}, resampling to {self.sampling_rate}")
            wave = ta.functional.resample(wave, sr, self.sampling_rate)

        # LUFS normalization to a given loudness
        wave_loudness = AF.loudness(wave, self.sampling_rate)
        gain = self.loudness - wave_loudness
        multiplier = 10 ** (gain/20)
        wave = wave * multiplier
        
        # for testing sample only seconds 10 to 20
        llimit = int(self.sampling_rate * 5)#10)
 
        wave = wave[:, int(llimit):int(llimit*2)]

        # most of the time the first and last 10 seconds should be cut off 
        # wave = wave[:, int(llimit):int(llimit*5)+1]

        # Squeeze to go from 1,x to x shape tensor
        wave = to_device(wave.squeeze(), self.device)
        return wave, sr, int(self.fileLabels[idx])
    
class SpectroDataLoader():
    """
    """

    def __init__(self, datas, batch_size, samples, device='cpu'):
        self.waves = []
        self.lbs = []
        self.sr = []
        self.device = device

        # Append to each array the values returned in the dataset class (wave, sampling_rate, label)
        for idx in samples:
            print(idx)
            elem=datas[idx]
            self.waves.append(elem[0])
            self.sr.append(elem[1])
            self.lbs.append(elem[-1])

        # Shuffle all waves
        self.datas = list(zip(self.waves, self.lbs))
        random.shuffle(self.datas)

        # Attributes for batching
        self.batch_size = batch_size
        self.image_batches = []
        self.label_batches = []

        while (len(self.datas) / self.batch_size > 0) | (len(self.datas) % self.batch_size != 0):

            # first n entries of shuffled vector are batch
            batch = self.datas[:self.batch_size]

            wvs = []
            las = []

            # Need to separate images and labels into two tensors
            for wv, la in batch:
                wvs.append(wv)
                las.append(la)
            del self.datas[:self.batch_size]

            le = LabelEncoder()
            lef = le.fit(las)
            las = lef.transform(las)

            self.image_batches.append(wvs)
            self.label_batches.append(las)

    def __iter__(self):
        bat = list(zip(self.image_batches, self.label_batches))
        random.shuffle(bat)

        # return a random batch
        for image_batch, label_batch in bat:
            yield to_device([torch.stack(image_batch), torch.Tensor(label_batch)], self.device)

    def __len__(self):
        return len(self.image_batches)