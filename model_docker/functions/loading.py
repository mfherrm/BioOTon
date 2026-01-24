#%% Data loading functions ###
import torch.nn as nn
import torchvision as tv

from functions.convenience import to_device
from functions.dataset import splitDataset, getSamples
from functions.network_components import AudioToLogSpectrogram, AudioToMelSpectrogram, AudioToMFCCSpectrogram
from functions.torch_classes import CombinedSpectroDataset, SpectroDataLoader

def load_dataset(dawn_file, xeno_file, augmented_file, splits = [0.7, 0.2], samples = [0.001,  0.001,  0.001], device = "cuda", denoised = True, cut = True, sample_recording = (False,15), sftp_config : dict = {}):
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
        ds = CombinedSpectroDataset(dawn_file, xeno_file, augmented_file, device = device, denoised = denoised, cut = cut, sample = sample_recording, sftp_config=sftp_config)
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