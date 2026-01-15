# from itertools import compress
import os
import pandas as pd
from pathlib import Path

import random
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
# import torchaudio as ta
# from torchcodec.decoders import AudioDecoder
# import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader

from functions.dl.convenience_functions import to_device
from functions.processing.retrieval import process_points_dir, process_cut_points_dir


class SpectroDataset(Dataset):
    """
    Custom dataset for bird sound data.

    Inputs: 
    
        recording_path : list[str] - path to a directory containing audio files in a .pt- format 
        label_path : list[str] - path to a directory containing a parquet-file with the labels 
        device : str - a string of a valid device to store the data on, e.g., cpu or cuda
        denoised : int - whether to use denoised data
        filtered : bool - whether to use the denoised and filtered data

    Methods:
        __init__ 
            instantiates the class and all relevant variables
            loads all paths for all sound waves that are both in the recording directory and the labels

        __len__ 
            returns the length of the dataset

        __getitem__
            returns an item at the specified index

    """ 
    def __init__(self, 
                recording_path: list[str], 
                label_path: list[str],
                #  transform: None | Callable,
                # sampling_rate: int = 44100, #Hz
                # loudness: int = 10,
                device :str = 'cpu',
                denoised : bool = False,
                filtered : bool = False,
                **kwargs
                ):
        # Assign instance variables
        self.l_path = label_path
        self.r_path = recording_path
        self.device = device
        self.denoised = denoised
        self.filtered = filtered

        # Load point labels
        dir_files = os.listdir(self.l_path)
        soundscape_file = next(f for f in dir_files if f.endswith("_single.parquet"))
        sound_df = pd.read_parquet(os.path.join(self.l_path, soundscape_file))
        sound_df = sound_df.drop(sound_df.loc[sound_df.label.isna()].index)

        if self.denoised:
            self.file_end = "_dn.pt"
        elif self.filtered:
            self.file_end = "_dn_bf.pt"
        else: 
            self.file_end = ".pt"

        # Load processed sound waves
        rec_ids = {int(f.split('_')[0]) for f in os.listdir(self.r_path) if f.endswith(str(self.file_end))}

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
        print("Getting", os.path.join(self.r_path, f"{self.fileNames[idx]}_audio{self.file_end}"))

        wave = torch.load(os.path.join(self.r_path, f"{self.fileNames[idx]}_audio{self.file_end}"))

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
  
    
class CombinedSpectroDataset(Dataset):
    """
    Custom dataset for bird sound data. This class implements data from multiple sources, i.e. Dawn Chorus, Xeno-Canto and data augmentation. 

    Inputs: 
    
        dawn_points_file : str - path to a parquet-file the Dawn Chorus labels
        xeno_points_file : str - path to a parquet-file the Xeno-Canto labels
        augmented_points_file : str - path to a parquet-file the augmented data labels
        device : str - a string of a valid device to store the data on, e.g., cpu or cuda
        denoised : bool - whether to use denoised data
        cut : tuple[bool, int] - whether to use use only a portion of the data and how many seconds

    Methods:
        __init__ 
            instantiates the class and all relevant variables
            loads all paths for all sound waves that are both in the recording directory and the labels for all data sources

        __len__ 
            returns the length of the dataset

        __getitem__
            returns an item at the specified index

    """ 
    def __init__(self, 
                dawn_points_file: str,
                xeno_points_file: str,
                augmented_points_file: str,
                device : str = 'cpu',
                denoised : bool = False,
                cut : bool = False,
                sample : tuple[bool, int] = (True, 5),
                **kwargs
                ):
        # Assign instance variables
        self.device = device
        self.sample = sample
        self.cut = cut

        self.denoised = denoised

        folder_suffix=""
        # Query denoised to get the file path to be used
        if self.denoised:
            folder_suffix += "_denoised"
        if self.cut:
            folder_suffix += "_cut"
            self.dawn_points_file_path, self.dawn_recordings_path, self.dawn_df, self.dawn_files, self.dawn_subset, self.dawn_filtered_files, self.dawn_filtered_labels = process_cut_points_dir(Path(dawn_points_file), "AudioTensors", folder_suffix)
            self.xeno_points_file_path, self.xeno_recordings_path, self.xeno_df, self.xeno_files, self.xeno_subset, self.xeno_filtered_files, self.xeno_filtered_labels = process_cut_points_dir(Path(xeno_points_file), "XenoCanto", folder_suffix)
            self.augmented_points_file_path, self.augmented_recordings_path, self.augmented_df, self.augmented_files, self.augmented_subset, self.augmented_filtered_files, self.augmented_filtered_labels = process_cut_points_dir(Path(augmented_points_file), "augmented_data", folder_suffix)
        
            self.combined_file_paths = [*self.dawn_filtered_files, *self.xeno_filtered_files, *self.augmented_filtered_files]
            self.combined_point_labels = [*self.dawn_filtered_labels, *self.xeno_filtered_labels, *self.augmented_filtered_labels]
        else: 
            self.dawn_points_file_path, self.dawn_recordings_path, self.dawn_df, self.dawn_files, self.dawn_subset, self.dawn_filtered = process_points_dir(Path(dawn_points_file), "AudioTensors", folder_suffix)
            self.xeno_points_file_path, self.xeno_recordings_path, self.xeno_df, self.xeno_files, self.xeno_subset, self.xeno_filtered = process_points_dir(Path(xeno_points_file), "XenoCanto", folder_suffix)
            self.augmented_points_file_path, self.augmented_recordings_path, self.augmented_df, self.augmented_files, self.augmented_subset, self.augmented_filtered = process_points_dir(Path(augmented_points_file), "augmented_data", folder_suffix)

            self.combined_file_paths = [*self.dawn_filtered, *self.xeno_filtered, *self.augmented_filtered]
            self.combined_point_labels = [*self.dawn_subset["label"], *self.xeno_subset["label"], *self.augmented_subset["label"]]

    
        label_encoder = LabelEncoder()
        self.combined_encoded_labels = label_encoder.fit_transform(self.combined_point_labels)

        self.datas = list(zip(self.combined_file_paths, self.combined_encoded_labels))
        random.shuffle(self.datas)

        if self.sample[0]:
            # for testing sample only seconds 5 to 10
            self.llimit = int(16000 * self.sample[1])
            self.rlimit = int(self.llimit * 2)


    def __len__(self):
        return len(self.combined_encoded_labels)
    
    def __getitem__(self, idx:int):
        print(f"Getting index {idx}, {self.datas[idx][0]}")

        path, label = self.datas[idx]

        wave = torch.load(path)

        if self.sample[0]:
            wave = wave[self.llimit:self.rlimit]

        return wave, int(label)
    

class SpectroDataLoader(DataLoader):
    """
    Custom dataloader to load a spectro-dataset to iterate it for training.

    Inputs: 
    
        data: str - a custom spectro-dataset
        batch_size - the size of data batches in the dataloader
        samples : list[int] - the subset of the dataset to be used
        device : str - a string of a valid device to store the data on, e.g., cpu or cuda

    Methods:
        __init__ 
            instantiates the class and all relevant variables

        __len__ 
            returns the number of batches in the dataloader

        __iter__
            iterates through the dataset, calculates a batch and returns it

    """ 
    def __init__(self, datas, batch_size, samples: list[int], sample_rate = 16000, clip_length = 60, device : str = 'cpu'):
        self.datas = datas
        self.batch_size = batch_size
        self.samples = samples
        self.device = device
        
        self.sample_rate = sample_rate
        self.wanted_recording_length = sample_rate * clip_length # seconds
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
                pad_size = self.wanted_recording_length - wave.shape[-1]

                if wave.ndim == 2:
                    wave = wave.squeeze(0)
                if pad_size > 0:
                    wave = F.pad(wave, (0, pad_size), mode='constant', value=0.0)
                elif pad_size < 0:
                    wave = wave[:self.wanted_recording_length]

                wvs.append(wave)
                las.append(label)

            yield to_device([torch.stack(wvs), torch.Tensor(las)], self.device)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size