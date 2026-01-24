import io
from pathlib import Path
import os
from ray import tune
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from functions.convenience import to_device
from functions.loading import load_dataset, load_data, load_model
from functions.network_components import EarlyStopping
from functions.sftp import get_pool
#%% Training functions ###
def train_model(config, sftp_config, dataset_files, spectro_mode="atls", device="cuda", split = [0.7, 0.1], samples = [0.01,  0.01,  0.01], clip_files = False, clip_length=15):
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
    pool = get_pool(**sftp_config)
    sftp = pool.get_sftp()
    # Load dataset
    # train_model, dataset_files=[dawn_file, xeno_file, augmented_file], spectro_mode = "atls", split = [0.7, 0.1], samples = [0.001,  0.001,  0.001], clip_files = False, clip_length=15
    ds, train_indices, val_indices = load_dataset(dataset_files[0], dataset_files[1], dataset_files[2], splits = split, samples = samples, device = device, denoised = True, cut = True, sample_recording = (clip_files,clip_length), sftp_config=sftp_config)
    print("INDICES", len(train_indices), len(val_indices))
    # Load dataloaders
    train_dataloader, val_dataloader = load_data(config, ds, split = [train_indices, val_indices], clip_length=clip_length)
    print("LENS", len(train_dataloader), len(val_dataloader))
    # Get the unique trial ID
    trial_id = tune.get_context().get_trial_id()

    nnw = load_model(config=config, mode = spectro_mode, device=device)

    writer = SummaryWriter(os.path.abspath("./runs/single_points"))

    networkPath = f"{Path(sftp.getcwd()).parent}/Masterarbeit_Marius/TrainingCheckpoints/checkpoint_{trial_id}.pt"#f"F:\\Persönliches\\Git\\BioOTon\\checkpoints\\checkpoint_{trial_id}.pt"
    checkpoint_dir = os.path.dirname(networkPath)
    print("CHECKPOINT DIR", checkpoint_dir)
    try:
        sftp.mkdir(checkpoint_dir)
    except IOError:
        print(f"Directory '{checkpoint_dir}' already exists.")
        pass
    loss = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=config["patience"], delta=config["EarlyDelta"], model = nnw, path = networkPath, sftp_config=sftp_config)

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

    buffer = io.BytesIO()
    torch.save(nnw.state_dict(), buffer)

    # Upload the buffer to the SFTP server
    buffer.seek(0)
    with sftp.open(str(networkPath), 'wb', bufsize=32768) as f:
        f.write(buffer.getbuffer())