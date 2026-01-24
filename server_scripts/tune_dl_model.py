#%% Imports ###
from collections import defaultdict
import getpass
import io

import numpy as np
import os
import pandas as pd
import paramiko
from pathlib import Path
import random
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
import re
from sklearn.preprocessing import LabelEncoder
import threading
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio as ta
import torchvision as tv
import torchvision.transforms as transforms

ray.init(ignore_reinit_error=True)
print("Ray is initialized!")

CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA = 1

thread_local = threading.local()
_pool = None


#%% Convenience functions ###
# Method to move tensors to chosen device
def to_device(data, device : str):
    """
        Moves tensors or models (pytorch data) to chosen device.

        Inputs:
            data - the pytorch data to be moved to the specified device

        Output:
            data - the data moved to the specified device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def trialDir(trial):
    """
        Used to set the ray[tune] runs to a trial id instead of the hyperparameters used.

        Input:
            trial - a ray[tune] trial

        Output:
            str - a formatted directory to save the trial to 
    """
    return f"/single_point/RayTune/{trial.trial_id[:6]}"
#%% SFTP functions ###
class SFTPConnectionPool:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.transport = None
        self._lock = threading.Lock()
        self.thread_local = threading.local()

    def _get_transport(self):
        with self._lock:
            # Check if transport is dead or closed
            if self.transport is None or not self.transport.is_active():
                print(f"[{threading.current_thread().name}] Establishing new Transport...")
                self.transport = paramiko.Transport((self.host, self.port))
                self.transport.connect(username=self.username, password=self.password)
            return self.transport

    def get_sftp(self):
        # If the thread already has a client, check if it's still alive
        if hasattr(self.thread_local, "sftp"):
            try:
                self.thread_local.sftp.listdir('.') # Test the connection
                return self.thread_local.sftp
            except:
                del self.thread_local.sftp # It's dead, remove it

        # Attempt to create a new SFTP client with retries
        for i in range(5): # Try 5 times to get a channel
            try:
                time.sleep(random.uniform(0, 32))
                # transport = self._get_transport()
                # sftp = paramiko.SFTPClient.from_transport(transport)
                transport = paramiko.Transport((self.host, self.port))
                transport.connect(username=self.username, password=self.password)
                sftp = paramiko.SFTPClient.from_transport(transport)
                absolute_start_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
                sftp.chdir(absolute_start_path)
                # self.thread_local.sftp = sftp
                self.thread_local.transport = transport 
                self.thread_local.sftp = sftp
                return sftp
            except Exception as e:
                wait = (i + 1) * 2
                print(f"Channel failed, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                # Force transport reset on 3rd failure
                if i == 2: self.transport = None 
        
        raise RuntimeError("Could not connect to SFTP after 5 attempts")

def get_pool():
    global _pool
    if _pool is None:
        # Import and initialize ONLY when called
        print("Worker: Initializing new Connection Pool...")
        _pool = SFTPConnectionPool(host, port, username, password)
    return _pool

#%% Data loading functions ###      
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
            
            pts = []
            wvs = []
            las = []


            for idx in batch_indices:
                index, wave, label = self.datas[idx]
                pad_size = self.wanted_recording_length - wave.shape[-1]

                if wave.ndim == 2:
                    wave = wave.squeeze(0)
                if pad_size > 0:
                    wave = F.pad(wave, (0, pad_size), mode='constant', value=0.0)
                elif pad_size < 0:
                    wave = wave[:self.wanted_recording_length]

                pts.append(index)
                wvs.append(wave)
                las.append(label)

            yield to_device([torch.Tensor(pts), torch.stack(wvs), torch.Tensor(las)], self.device)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def load_dataset(dawn_file, xeno_file, augmented_file, splits = [0.7, 0.2], samples = [0.001,  0.001,  0.001], device = "cuda", denoised = True, cut = True, sample_recording = (False,15)):
        # The train - test - validation split
        train_split_pct = splits[0]
        test_split_pct = splits[1]
        val_split_pct = 1.0 - train_split_pct - test_split_pct

        # If and how many samples to take from the dataset
        train_sample_size = samples[0]
        test_sample_size = samples[1]
        val_sample_size = samples[2]

        # Load  dataset
        # ds = SpectroDataset(recording_path, label_path, device = 'cuda', denoised = True, filtered=False)
        ds = CombinedSpectroDataset(dawn_file, xeno_file, augmented_file, device = device, denoised = denoised, cut = cut, sample = sample_recording)
        print("Got Dataset")

        # Get samples and split the dataset
        split = splitDataset(ds, test_split_size=val_split_pct, val_split_size=test_split_pct)

        train_indices, test_indices, val_indices = getSamples(split, sizes = [train_sample_size, test_sample_size, val_sample_size])

        return ds, train_indices, val_indices


def load_data(config : dict, dataset, split : list = [0.7, 0.1], sample_rate : int = 16000, clip_length : int = 15):
    """
        Creates dataloaders for training and validation. Used in the train_model-function of the ray[tune] pipeline.

        Inputs: 
        
            config : dict - a dictionary made up of ray[tune] components specifying  hyperparameter bounds
            dataset - a pytorch dataset with a len() method 
            split : list - dataset split train / test / validation

        Outputs:

            train_dataloader  - a (custom) pytorch dataloader loaded with the training data
            val_dataloader - a (custom) pytorch dataloader loaded with the validation data
    """

    # Get training and validation data
    train_dataloader = SpectroDataLoader(dataset, config["batch_size"], samples = split[0], sample_rate = sample_rate, clip_length = clip_length, device = "cuda")
    val_dataloader = SpectroDataLoader(dataset, config["batch_size"], samples = split[1], sample_rate = sample_rate, clip_length = clip_length, device = "cuda")

    return train_dataloader, val_dataloader


def load_model(config, mode : str = "atls", device : str = "cuda"):
    """
        Creates and loads a model by combining a pretrained model with a spectrogram encoder.

        Inputs: 
        
            config : dict - a dictionary made up of ray[tune] components specifying  hyperparameter bounds
            mode : str - a string of the spectrogram encoder used, i.e., atls, atms or atmfs 
            device : str - a string of a valid device to store the data on, e.g., cpu or cuda 

        Outputs:

            nnw  - the resulting model loaded on the specified device
    """

    # Adapt model structure to in and output
    res = tv.models.resnet18()
    adaptconv1 = nn.Conv2d (in_channels=1, kernel_size=res.conv1.kernel_size, stride=res.conv1.stride, padding = res.conv1.padding, bias=res.conv1.bias, out_channels=res.conv1.out_channels)
    res.conv1 = adaptconv1
    res.fc = nn.Linear(in_features=res.fc.in_features, out_features=37, bias=True)

    # Initialize custom spectrogram module
    # Combine module and model and move it to the device
    nnw = None
    if mode == "atls":
        atls = AudioToLogSpectrogram(n_fft=config["nfft"], power = config["power"], device = device)
        nnw = nn.Sequential(atls, res)
    elif mode == "atms":
        atms = AudioToMelSpectrogram(n_fft=config["nfft"], n_mels=config["nmels"], device = device)
        nnw = nn.Sequential(atms, res)
    elif mode == "atmfs":
        atms = AudioToMFCCSpectrogram(n_mfcc= config["nmfcc"], n_fft=config["nfft"], n_mels=config["nmels"], device = device)
        nnw = nn.Sequential(atms, res)

    return to_device(nnw, device)


#%% Dataset retrieval functions ###
def process_points_dir(file_path, folder, folder_suffix):
    """
    Processes a file path, folder and corresponding suffix and returns recording locations and the corresponding labels. Intended for use in the CombinedSpectroDataset.

    Input:
        file_path - path to a .pt-file containing point labels
        folder - folder NAME where the recordings are located in
        folder_suffix - to enable loading both normal and denoised data 

    Output: 
        file_path - path to the label file
        recordings_path - path to the directory containing the recordings
        point_df - the label dataframe
        audio_files - paths to all audio files
        point_subset - point_df filtered to only include points that are in the file directory
        filtered_audio_files - audio_files filtered to include only files that occur in point_subset
    """
    recordings_path = Path(f"{file_path.parent}/{folder}{folder_suffix}")

    # Read all label points
    point_df = pd.read_parquet(file_path)

    # Get list of all recordings
    audio_files = list(recordings_path.glob("*.pt"))

    # Get the id as an integer
    rec_ids = {int(re.search(r"\d+", f.stem).group()) for f in audio_files}

    # Filter the label frame to only use the audio_files
    point_subset = point_df[point_df['id'].isin(rec_ids)].dropna(subset=['label'])

    # In some cases there are files in the recording directory that do not occur in the data frame
    # Thus, the file directory needs to be filtered to the dataframe as well
    # Map all integer ids to the corresponding file_path
    path_map = {int(re.search(r"\d+", p.stem).group()): p for p in audio_files}

    filtered_audio_files = [path_map[uid] for uid in point_subset['id'] if uid in path_map]


    return file_path, recordings_path, point_df, audio_files, point_subset, filtered_audio_files

def process_cut_points_dir(file_path, folder, folder_suffix, ignore_rem : bool = True):
    """
        Processes a file path, folder and corresponding suffix and returns recording locations and the corresponding labels. Intended for use in the CombinedSpectroDataset.

        Input:
            file_path - path to a .pt-file containing point labels
            folder - folder NAME where the recordings are located in
            folder_suffix - to enable loading both normal and denoised data 

        Output: 
            file_path - path to the label file
            recordings_path - path to the directory containing the recordings
            point_df - the label dataframe
            audio_files - paths to all audio files
            point_subset - point_df filtered to only include points that are in the file directory
            filtered_audio_files - audio_files filtered to include only files that occur in point_subset
    """
    pool = get_pool()
    sftp = pool.get_sftp()
    recordings_path = Path(f"{file_path.parent}/{folder}{folder_suffix}")
    # Read all label points
    with sftp.open(file_path.as_posix(), 'rb') as remote_file:
        print(f"Opened label file {file_path}.")
        point_df = pd.read_parquet(remote_file)
        point_df = point_df[~point_df.label.isna()]
        point_df.id= point_df.id.astype(int)

    # Get list of all recordings
    #audio_files = list(recordings_path.glob("*.pt"))
    audio_files = sftp.listdir(recordings_path.as_posix())
    audio_files = [Path(file) for file in audio_files]

    audio_files_no_rem = [p for p in audio_files if "rem" not in p.stem]

    # Get the id as an integer
    rec_ids = {
        int(re.search(r'\d+', (s := f.name.split("_"))[3 if len(s) >= 6 else 2]).group())
        for f in audio_files_no_rem
    }

    # Filter the label frame to only use the audio_files
    point_subset = point_df[point_df['id'].isin(rec_ids)].dropna(subset=['label'])

    # In some cases there are files in the recording directory that do not occur in the data frame
    # Thus, the file directory needs to be filtered to the dataframe as well
    # Map all integer ids to the corresponding file_path
    path_map = defaultdict(list)
    for p in audio_files_no_rem:
        uid = int(re.search(r'\d+', (s := p.name.split("_"))[3 if len(s) >= 6 else 2]).group())
        path_map[uid].append(p)

    filtered_path_map = {uid:path_map[uid] for uid in point_subset['id'] if uid in path_map}
        
    filtered_audio_files= []
    filtered_audio_labels = []

    for id in filtered_path_map:
        label = int(point_subset[point_subset["id"].isin([id])].label.unique().item())
        data = filtered_path_map[id]
        cuts = len(data)
 
        for i in range(cuts):
            filtered_audio_files.append(f"{sftp.getcwd()}/{folder}{folder_suffix}/{data[i]}")
            filtered_audio_labels.append(label)


    return file_path, recordings_path, point_df, audio_files, point_subset, filtered_audio_files, filtered_audio_labels

#%% Dataset functions ###
def splitDataset(dataset, test_split_size : float = 0.2, val_split_size : float = 0.1):
    """
        Splits a dataset into training, test and validation indices.

        Inputs: 

            dataset - a pytorch dataset with a len() method 
            test_split_size : float = 0.2 - portion of the dataset to be used for testing
            val_split_size : float = 0.1 - portion of the dataset to be used for validation

        Outputs:

            train_indices : list[int] - list of indices for training
            test_indices : list[int] - list of indices for testing
            val_indices : list[int] - list of indices for validation 
    """

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    test_split = int(np.floor(test_split_size * dataset_size))
    val_split = int(np.floor(val_split_size * dataset_size))

    val_indices, test_indices, train_indices = indices[:val_split], indices[val_split: (val_split+test_split)], indices[(val_split+test_split):]

    return train_indices, test_indices, val_indices 


def getSamples(split : list, sizes : list = [None, None, None]):
    """
        Computes the indices for the datasets used in the dataloader. Intended to be used after the splitDataset method.

        Inputs:
            split : list - list containing indices of the train, test and val splits
            sizes : list - list containing the number of samples. Uses absolute samples when given an int, percentage for float <= 1 and all sample for None type

        Outputs: 
            samples : list - list containing the split indices
    """
    samples = []
    for idx, data in enumerate(split):
        data_indices = []
        if type(sizes[idx]) is int:
            data_indices = split[idx][:sizes[idx]]
        elif (type(sizes[idx]) is float and sizes[idx] <= 1.0):
            indices = round(len(split[idx])*sizes[idx])
            data_indices = split[idx][:indices]
        elif sizes[idx] is None:
            data_indices = split[idx]
        
        samples.append(data_indices)
        
    return samples


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
                aggregate_labels : bool = False,
                sample : tuple[bool, int] = (True, 5),
                **kwargs
                ):
        # Assign instance variables
        self.device = device
        self.denoised = denoised
        self.cut = cut
        self.aggregate_labels = aggregate_labels
        self.sample = sample

        

        folder_suffix = ""
        # Query denoised to get the file path to be used
        if self.denoised:
            folder_suffix += "_denoised"
        if self.cut:
            folder_suffix += "_cut"
            self.dawn_points_file_path, self.dawn_recordings_path, self.dawn_df, self.dawn_files, self.dawn_subset, self.dawn_filtered_files, self.dawn_filtered_labels = process_cut_points_dir(Path(dawn_points_file), "Dawn", folder_suffix)
            self.xeno_points_file_path, self.xeno_recordings_path, self.xeno_df, self.xeno_files, self.xeno_subset, self.xeno_filtered_files, self.xeno_filtered_labels = process_cut_points_dir(Path(xeno_points_file), "XenoCanto", folder_suffix)
            self.augmented_points_file_path, self.augmented_recordings_path, self.augmented_df, self.augmented_files, self.augmented_subset, self.augmented_filtered_files, self.augmented_filtered_labels = process_cut_points_dir(Path(augmented_points_file), "Augmented_data", folder_suffix)
        
            self.combined_file_paths = [*self.dawn_filtered_files, *self.xeno_filtered_files, *self.augmented_filtered_files]
            self.combined_point_labels = [*self.dawn_filtered_labels, *self.xeno_filtered_labels, *self.augmented_filtered_labels]
        else: 
            self.dawn_points_file_path, self.dawn_recordings_path, self.dawn_df, self.dawn_files, self.dawn_subset, self.dawn_filtered = process_points_dir(Path(dawn_points_file), "Dawn", folder_suffix)
            self.xeno_points_file_path, self.xeno_recordings_path, self.xeno_df, self.xeno_files, self.xeno_subset, self.xeno_filtered = process_points_dir(Path(xeno_points_file), "XenoCanto", folder_suffix)
            self.augmented_points_file_path, self.augmented_recordings_path, self.augmented_df, self.augmented_files, self.augmented_subset, self.augmented_filtered = process_points_dir(Path(augmented_points_file), "Augmented_data", folder_suffix)

            self.combined_file_paths = [*self.dawn_filtered, *self.xeno_filtered, *self.augmented_filtered]
            self.combined_point_labels = [*self.dawn_subset["label"], *self.xeno_subset["label"], *self.augmented_subset["label"]]
        if self.aggregate_labels:
            self.combined_point_labels = [int(label/10) for label in self.combined_point_labels]
    
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

        pool = get_pool()
        sftp = pool.get_sftp()

        with sftp.open(path, 'rb') as remote_file:
            file_data = remote_file.read()
            buffer = io.BytesIO(file_data)
            try:
                wave = torch.load(buffer, map_location=device)
            except Exception as e:
                print(f"\n[CRITICAL CORRUPTION] File {path} is unreadable: {e}")

        if self.sample[0]:
            wave = wave[self.llimit:self.rlimit]

        return int(idx), wave, int(label)


#%% Network components ###
# https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
class EarlyStopping:
    """
        Class used in train model to perform early stoppping.

        Inputs: 
    
            model - the model that is being trained
            patience : int - how many iterations without a decrease in validation loss until the training is stopped
            delta : float - bounds for which result will be considered no improvement
            window : int - how many iterations to consider for the delta
            path : str - path to the checkpoint directory
            verbose : bool - whether to communication decisions made

        Methods:
            __init__ 
                instantiates the class and all relevant variables
        
            check_early_stop
                computes the window delta
                saves the model upon improvement
                stops early if patience is reached
    """

    def __init__(self, model, patience : int = 5, delta : float = 0.001, window : int = 5, path : str ='checkpoints/checkpoint.pt', verbose : bool = True):
        self.patience = patience
        self.delta = delta
        self.window = window
        self.values = []
        self.path = path
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.model = model
    
    def check_early_stop(self, val_loss):
        if (len(self.values) < self.window):
            self.values.append(val_loss)
            return False
        else:
            last_value = np.std(self.values)
            
            sliced_values = self.values[1:]
            sliced_values.append(val_loss)
            
            current_value = np.std(sliced_values)
            
            self.values = sliced_values
        if current_value<last_value:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            # Save checkpoint if improvement observed
            torch.save(self.model.state_dict(), self.path)
            if self.verbose:
                print(f"Model improved; checkpoint saved at loss {val_loss:.4f}")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                print("Early stopping triggered.")
                return True  # Signal to stop training
        return False
    

class AudioToLogSpectrogram(torch.nn.Module):
    """
        Class used to compute log spectrograms.

        Inputs: 
    
            n_fft : int - number of fft bins
            power : float - Exponent for the magnitude spectrogram, (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            device : str - a string of a valid device to store the data on, e.g., cpu or cuda 

        Methods:
            __init__ 
                instantiates the class and all relevant variables
        
            forwards
                computes the spectrogram
    """
    def __init__(
        self,
        n_fft : int = 4096,
        power : float = 2.0,
        sample_rate = 16000,
        device : str = "cpu"
    ):
        super().__init__()
        
        self.spec = to_device(ta.transforms.Spectrogram(n_fft=n_fft, hop_length=n_fft//4, power=power), device)
        self.amplitude_to_db = ta.transforms.AmplitudeToDB(stype='power')
        # self.sample_rate = sample_rate
        # self.wanted_recording_length = self.sample_rate * 60 # seconds

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample
        # waveform = ta.transforms.resample()

        # Pad waveform to 60s run-time
        # pad_size = self.wanted_recording_length - waveform.shape[-1]
        # if pad_size > 0:
        #     waveform = F.pad(waveform, (0, pad_size), mode='constant', value=0.0)

        # Convert to power spectrogram
        spec = self.spec(waveform)

        spec_db = self.amplitude_to_db(spec)

        if spec_db.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            spec_db = spec_db.unsqueeze(0).unsqueeze(0)
        elif spec_db.ndim == 3:
            # [Batch, H, W] -> [Batch, 1, H, W]
            spec_db = spec_db.unsqueeze(1)

        # Z-score standardization
        # mean = spec_db.mean()
        # std = spec_db.std()
        # spec_db = (spec_db - mean) / (std + 1e-6)

        # Min-max normalization
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)

        
        # spec = torch.where(spec < 1, 1, spec)
        # spectrogram = torch.log10(spec) / self.scale
        # im = transforms.Resize((224, 224))(spectrogram[None, :, :]).squeeze()
        # return im.unsqueeze(1)
        # int(spec_db.shape[2]/8)
        im = transforms.Resize((int(spec_db.shape[2]/4), int(spec_db.shape[3]/4)))(spec_db)
        
        return im
    
class AudioToMelSpectrogram(torch.nn.Module):
    """
        Class used to compute Mel spectrograms.

        Inputs: 
            fmin : minimum frequency
            sample_rate : int - Sample rate of audio signal
            n_mels : int - Number of mel filterbanks
            n_fft : int - number of fft bins
            device : str - a string of a valid device to store the data on, e.g., cpu or cuda 

        Methods:
            __init__ 
                instantiates the class and all relevant variables
        
            forwards
                computes the spectrogram
    """
    def __init__(
        self,
        fmin : float = 0.0,
        sample_rate : int = 16000,
        n_mels : int = 128,
        n_fft=4096,
        device="cpu"
        
    ):
        super().__init__()
        # Nyquist theorem
        fmax = sample_rate/2

        self.mel_scale = to_device(ta.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft, 
            hop_length=n_fft//4,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels
        ), device)

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(stype='power')
        # self.sample_rate = sample_rate
        # self.wanted_recording_length = self.sample_rate * 60 # seconds

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to power spectrogram
        # pad_size = self.wanted_recording_length - waveform.shape[-1]
        # if pad_size > 0:
        #     waveform = F.pad(waveform, (0, pad_size), mode='constant', value=0.0)

        spec = self.mel_scale(waveform)

        spec_db = self.amplitude_to_db(spec)

        if spec_db.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            spec_db = spec_db.unsqueeze(0).unsqueeze(0)
        elif spec_db.ndim == 3:
            # [Batch, H, W] -> [Batch, 1, H, W]
            spec_db = spec_db.unsqueeze(1)

        # Z-score standardization
        # mean = spec_db.mean()
        # std = spec_db.std()
        # spec_db = (spec_db - mean) / (std + 1e-6)

        # Min-max normalization
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)

        im = transforms.Resize((int(spec_db.shape[2]/4), int(spec_db.shape[3]/4)))(spec_db)

        return im
    
class AudioToMFCCSpectrogram(torch.nn.Module):
    """
        Class used to compute Mel-frequency cepstrum coefficients.

        Inputs: 
            n_mfcc : number of mfc coefficients to retain
            n_fft : int - number of fft bins
            n_mels : int - Number of mel filterbanks
            sample_rate : int - Sample rate of audio signal
            device : str - a string of a valid device to store the data on, e.g., cpu or cuda 

        Methods:
            __init__ 
                instantiates the class and all relevant variables
        
            forwards
                computes the coefficients
    """
    def __init__(
        self,
        n_mfcc : int = 13,
        n_fft : int = 4096,
        n_mels : int = 23,
        sample_rate : int =  16000,
        device : str = "cpu"
    ):
        super().__init__()
        if n_mfcc > n_mels:
            print("Number of MFCC bins cannot be greater than number of Mel bins. Changing to a smaller number.")
            n_mfcc = n_mels - n_mels*0.1

        hop_length = n_fft//4
        
        self.transform = ta.transforms.MFCC(
            sample_rate = sample_rate,
            n_mfcc = n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "center": False},
        )
        # self.amplitude_to_db = ta.transforms.AmplitudeToDB(stype='power')

        self.device = device

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to mfcc spectrogram
        spec_db =  self.transform(waveform)

        # spec_db = self.amplitude_to_db(spec)

        if spec_db.ndim == 2:
            # [H, W] -> [1, 1, H, W]
            spec_db = spec_db.unsqueeze(0).unsqueeze(0)
        elif spec_db.ndim == 3:
            # [Batch, H, W] -> [Batch, 1, H, W]
            spec_db = spec_db.unsqueeze(1)

        # Z-score standardization
        # mean = spec_db.mean()
        # std = spec_db.std()
        # spec_db = (spec_db - mean) / (std + 1e-6)

        # Min-max normalization
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-6)

        im = transforms.Resize((int(spec_db.shape[2]/4), int(spec_db.shape[3]/4)))(spec_db)
        
        return to_device(im, self.device)
    

#%% Training functions ###
def train_model(config, dataset_files, spectro_mode="atls", device="cuda", split = [0.7, 0.1], samples = [0.01,  0.01,  0.01], clip_files = False, clip_length=15):
    """
    Train a model using ray[tune].

    Inputs: 
    
        config : dict - a dictionary made up of ray[tune] components specifying  hyperparameter bounds
        dataset - a pytorch dataset with a len() method
        mode : str - a string of the spectrogram encoder used, i.e., atls, atms or atmfs
        device : str - a string of a valid device to store the data on, e.g., cpu or cuda
        train_size : int - Number of samples used for training
        val_size : int - Number of samples used for validation

    Outputs:

    """ 
    pool = get_pool()
    # Load dataset
    # train_model, dataset_files=[dawn_file, xeno_file, augmented_file], spectro_mode = "atls", split = [0.7, 0.1], samples = [0.001,  0.001,  0.001], clip_files = False, clip_length=15
    ds, train_indices, val_indices = load_dataset(dataset_files[0], dataset_files[1], dataset_files[2], splits = split, samples = samples, device = device, denoised = True, cut = True, sample_recording = (clip_files,clip_length))
    print("INDICES", len(train_indices), len(val_indices))
    # Load dataloaders
    train_dataloader, val_dataloader = load_data(config, ds, split = [train_indices, val_indices], clip_length=clip_length)
    print("LENS", len(train_dataloader), len(val_dataloader))
    # Get the unique trial ID
    trial_id = tune.get_context().get_trial_id()

    nnw = load_model(config=config, mode = spectro_mode, device=device)

    writer = SummaryWriter("runs/single_points")

    networkPath = f"{pool.get_sftp().getcwd()}/checkpoints/checkpoint_{trial_id}.pt"#f"F:\\Persönliches\\Git\\BioOTon\\checkpoints\\checkpoint_{trial_id}.pt"
    loss = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["patience"], delta=config["EarlyDelta"], model = nnw, path = networkPath)

    # fused doesn't work without cuda
    if device == 'cuda':
        optimizer = optim.Adam(nnw.parameters(), lr=config["lr"], fused=True)
    else:
        optimizer = optim.Adam(nnw.parameters(), lr=config["lr"])

    # Train the network
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        nnw.train()
        running_loss = 0.0
        # i = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            _, inputs, labels = data

            outputs = nnw(inputs)

            labels_long = labels.type(torch.LongTensor)
            labels_long = to_device(labels_long.long(), device)
            # print(f"Batch Label Min: {labels_long.min()}, Max: {labels_long.max()}")

            los = loss(outputs, labels_long)


            l1_penalty = 0
            l2_penalty = 0

            # L1 and L 2Regularization
            for p in nnw.parameters():
                l1_penalty += p.abs().sum()
                l2_penalty += p.pow(2.0).sum()

            # Elastic Net Penalty
            elastic_penalty = config["l1"] * l1_penalty + config["l2"] * l2_penalty

            los += elastic_penalty

            # Clear the gradients
            optimizer.zero_grad()

            # Backpropagation to compute gradients
            los.backward()

            # Update model parameters
            optimizer.step()

            # print statistics
            running_loss += los.item()

            print(config["batch_size"], " : ", i % config["batch_size"])

        avg_loss = 0.0
        avg_vloss = 0.0
        total = 0
        correct = 0

        

        
        # Check against validation dataset
        running_vloss = 0.0

        # Switch to evaluation mode to omit some model specific operations like dropout
        nnw.train(False)
        for j, vdata in enumerate(val_dataloader, 0): 
            _, vinputs, vlabels = vdata
            vlabels_long = to_device(vlabels.type(torch.LongTensor), device)

            voutputs = nnw(vinputs)
            _, predicted = torch.max(voutputs.data, 1)
            total += vlabels_long.size(0)
            correct += (predicted == vlabels_long).sum().item()

            vloss = loss(voutputs, vlabels_long)
            running_vloss  +=vloss.item()
        
        nnw.train(True)
        avg_loss = running_loss / config["batch_size"]

        avg_vloss = running_vloss / len(val_dataloader)

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch * len(train_dataloader) + i)

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.6f} vloss: {avg_vloss:.6f}')

        running_loss = 0.0

        tune.report({"loss": avg_vloss, "accuracy": correct/total})
            
            
        if early_stopping.check_early_stop(avg_vloss):
            print(f"Stopping training at epoch {epoch+1}")
            break

    nnw.eval()
    print('Finished Training')

    writer.flush()
    print("Flushed writer")

    torch.save(nnw.state_dict(), networkPath)

#%% Execution ###
if __name__ == '__main__':
    ### SFTP configurations ###
    host = 'os-login.lsdf.kit.edu'
    port = 22

    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    # pool = SFTPConnectionPool(host, port, username , password)

    #### Configurations ####

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # https://medium.com/biased-algorithms/hyperparameter-tuning-with-ray-tune-pytorch-d5749acb314b
    max_epochs = 25
    # Define hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-5, 1e-2),          # Learning rate between 1e-5 and 1e-2
        # "batch_size": tune.lograndint(8, 16), # for small batch testing
        "batch_size": tune.lograndint(32, 128),
        "nfft" : tune.choice([512, 1024, 2048, 4096, 8192]),
        # "scale" : tune.uniform(0.5, 4),
        "power" : tune.uniform(0.5, 4),
        "patience" : tune.choice([2, 3, 5, 7, 9]),
        "EarlyDelta" : tune.uniform(0.0015, 0.1),
        "epochs" : tune.randint(15, max_epochs),
        "l1" : tune.loguniform(0.0005, 0.004),
        "l2" : tune.loguniform(0.00075, 0.003),
        "nmels" : tune.randint(64, 256),
        "nmfcc" : tune.randint(32, 128)
        #"optimizer": tune.choice(["adam", "sgd"]),  # Optimizer choice: Adam or SGD
        # "layer_size": tune.randint(64, 256),        # Random integer for layer size (hidden units)
        # "dropout_rate": tune.uniform(0.1, 0.5)      # Dropout rate between 0.1 and 0.5
    }

    # Rayt[tune] parameters
    concurrent_trials = 4
    
    # File locations for the label files
    base_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
    dawn_file = Path(f"{base_path}/points_single.parquet") # F:/Persönliches/Git/BioOTon/points_single.parquet"
    xeno_file = Path(f"{base_path}/xeno_points_single.parquet") # "F:/Persönliches/Git/BioOTon/xeno_points_single.parquet"
    augmented_file = Path(f"{base_path}/augmented_points_single.parquet") # "F:/Persönliches/Git/BioOTon/augmented_points_single.parquet"

    trainable_with_parameters = tune.with_parameters(
        train_model, dataset_files=[dawn_file, xeno_file, augmented_file], spectro_mode = "atls", split = [0.7, 0.1], samples = [0.001,  0.001,  0.001], clip_files = False, clip_length=15 #train_size = int(np.floor(0.7 * len(ds))), val_size = int(np.floor(0.1 * len(ds)))# train_size=500, val_size=100
    )

    cpu_count = os.cpu_count()

    trainable_with_resources = tune.with_resources(
        trainable_with_parameters,
        resources={"cpu": int(cpu_count/2)/concurrent_trials, "gpu": 1/concurrent_trials, "accelerator_type:G":1/concurrent_trials}
)

optuna_search = OptunaSearch(
    metric=["loss", "accuracy"],
    mode=["min", "max"]
)

# Currently unused
hyperopt_search = HyperOptSearch(
    metric="loss",
    mode="min",  # Minimize loss
    # points_to_evaluate # Use when some good hyperparameters are known as initial values
)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    max_t=max_epochs,
    grace_period=10,
    brackets=1,
)

# Pass the search algorithm to Ray Tune
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=config,
    # tune_config=tune.TuneConfig(search_alg=hyperopt_search, num_samples=50, trial_dirname_creator=trialDir, max_concurrent_trials=2),
    tune_config=tune.TuneConfig(search_alg=optuna_search, num_samples=50, trial_dirname_creator=trialDir, max_concurrent_trials=concurrent_trials,),
    run_config=tune.RunConfig(storage_path='/RayResults', name="results")
)
tuner.fit()