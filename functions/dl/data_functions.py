from itertools import compress
import numpy as np
import os
from ray import tune
from ray.tune import ExperimentAnalysis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv

from functions.dl.data_classes import SpectroDataLoader
from functions.dl.network_components import AudioToLogSpectrogram, AudioToMelSpectrogram, AudioToMFCCSpectrogram, EarlyStopping
from functions.dl.convenience_functions import to_device


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


def load_data(config : dict, dataset, split : list = [0.7, 0.2, 0.1], sample_rate : int = 16000, clip_length : int = 15):
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


def train_model(config, dataset, spectro_mode="atls", device="cuda", split = [0.7, 0.2, 0.1], clip_length=15):
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
    # Load data
    train_dataloader, val_dataloader = load_data(config, dataset, split = split, clip_length=clip_length)

    # Get the unique trial ID
    trial_id = tune.get_context().get_trial_id()


    nnw = load_model(config=config, mode = spectro_mode, device=device)

    writer = SummaryWriter("runs/single_points")

    networkPath = f"F:\\Persönliches\\Git\\BioOTon\\checkpoints\\checkpoint_{trial_id}.pt"
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

        avg_loss = 0.0
        avg_vloss = 0.0
        total = 0
        correct = 0

        print(config["batch_size"], " : ", i % config["batch_size"])

        
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


def getBestModel(path="D:\\ProgramFiles\\RayResults\\results", metric : str = "loss", mode : str = "min", return_df : bool = False, device="cpu"):
    """
    Queries completed ray tune runs and returns the best model.

    Inputs: 

        path - path to the ray[tune] results directory.
        metric : str - metric by which the quality of a model is measured 
        mode : str - whether the metric should by as low or high as possible, i.e. min or max
        return_df : bool - whether to return the dataframe with the results
        device : str - a string of a valid device to store the data on, e.g., cpu or cuda

    Outputs:

        model  - the resulting model loaded on the specified device
        df - dataframe with the results
    """   
    analysis = ExperimentAnalysis(path)

    # Get the best hyperparameters based on loss
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    print("Best Hyperparameters:", best_config)

    # Get the best trial based on the metric
    best_result = analysis.get_best_trial(metric=metric, mode=mode)

    df = analysis.results_df

    # Load model
    model = load_model(best_config)

    # Get list of checkpoints and find the file that matches the best result
    dir_files = os.listdir("./checkpoints")
    model_location = list(compress(dir_files, [file.endswith(str(best_result).split("_")[-1]+".pt") for file in dir_files]))[0]

    # load model state
    keys = model.load_state_dict(torch.load(f"checkpoints/{model_location}"))
    if return_df:
        return to_device(model, device), df
    else: 
        return to_device(model, device)