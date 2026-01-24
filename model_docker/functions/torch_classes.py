import io
from pathlib import Path
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from functions.convenience import to_device
from functions.retrieval import process_points_dir, process_cut_points_dir
from functions.sftp import get_pool

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
                sftp_config: dict = {},
                **kwargs
                ):
        # Assign instance variables
        self.device = device
        self.denoised = denoised
        self.cut = cut
        self.aggregate_labels = aggregate_labels
        self.sample = sample
        self.sftp_config = sftp_config

        

        folder_suffix = ""
        # Query denoised to get the file path to be used
        if self.denoised:
            folder_suffix += "_denoised"
        if self.cut:
            folder_suffix += "_cut"
            self.dawn_points_file_path, self.dawn_recordings_path, self.dawn_df, self.dawn_files, self.dawn_subset, self.dawn_filtered_files, self.dawn_filtered_labels = process_cut_points_dir(Path(dawn_points_file), "Dawn", folder_suffix, sftp_config=sftp_config)
            self.xeno_points_file_path, self.xeno_recordings_path, self.xeno_df, self.xeno_files, self.xeno_subset, self.xeno_filtered_files, self.xeno_filtered_labels = process_cut_points_dir(Path(xeno_points_file), "XenoCanto", folder_suffix, sftp_config=sftp_config)
            self.augmented_points_file_path, self.augmented_recordings_path, self.augmented_df, self.augmented_files, self.augmented_subset, self.augmented_filtered_files, self.augmented_filtered_labels = process_cut_points_dir(Path(augmented_points_file), "Augmented_data", folder_suffix, sftp_config=sftp_config)
        
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

        pool = get_pool(self.sftp_config)
        sftp = pool.get_sftp()

        with sftp.open(path, 'rb') as remote_file:
            file_data = remote_file.read()
            buffer = io.BytesIO(file_data)
            try:
                wave = torch.load(buffer, map_location=self.device)
            except Exception as e:
                print(f"\n[CRITICAL CORRUPTION] File {path} is unreadable: {e}")

        if self.sample[0]:
            wave = wave[self.llimit:self.rlimit]

        return int(idx), wave, int(label)


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