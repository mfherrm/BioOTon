import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from functions.dl.data_classes import SpectroDataset, SpectroDataLoader
from functions.dl.network_components import EarlyStopping, AudioToLogSpectrogram, AudioToMelSpectrogram
from functions.dl.data_functions import splitDataset, load_data, load_model, train_model, getBestModel


def plot_spectrograms_dual(audio_files, dataset_dir = "./AudioTensors"):
    fig = plt.figure(figsize=(20, 12))
    
    # 3 rows, 7 columns
    # Column index 3 is the "divider" for the colorbar
    gs = gridspec.GridSpec(3, 7, figure=fig, 
                           width_ratios=[1, 1, 1, 0.4, 1, 1, 1],
                           wspace=0.3, hspace=0.3)

    fig.suptitle('Log-Mel Spectrogram Comparison', fontsize=20, y=0.95)

    atls = AudioToLogSpectrogram()
    atms = AudioToMelSpectrogram()

    ds = SpectroDataset(dataset_dir, os.getcwd(), device = 'cpu')
    train_indices, test_indices, val_indices  = splitDataset(ds)
    train_dataloader = SpectroDataLoader(ds, 128, samples= train_indices[:9], device = "cpu")
    waves = next(iter(train_dataloader))

    for i in range(9):
        if i >= len(audio_files): 
            break
        
        wav = waves[0][i]

        img_tensor = atls(wav).squeeze(0).squeeze(0)
        img = img_tensor.detach().cpu().numpy()

        row = i // 3
        col = i % 3

        # left
        ax_l = fig.add_subplot(gs[row, col])
        im = ax_l.imshow(img, aspect='auto', origin='lower', cmap='viridis')
        ax_l.axis('off')

        img_tensor = atms(wav).squeeze(0).squeeze(0)
        img = img_tensor.detach().cpu().numpy()

        # right
        ax_r = fig.add_subplot(gs[row, col + 4])
        ax_r.imshow(img, aspect='auto', origin='lower', cmap='viridis')
        ax_r.axis('off')

    # Set colorbar to middle of the plot
    cbar_ax = fig.add_subplot(gs[:, 3])
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 + 0.01, pos.y0 + 0.2, 0.015, pos.height - 0.3])
    
    cbar = fig.colorbar(im, cax=cbar_ax)
    
    # Set the label below colorbar
    cbar.ax.set_xlabel('Magnitude (dB)', fontsize=12, labelpad=15)
    
    plt.show()