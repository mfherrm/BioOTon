import torch
import numpy as np
import torchvision.transforms as transforms
import torchaudio as ta

from functions.dl.convenience_functions import to_device

# https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
class EarlyStopping:
    def __init__(self, model, patience=5, delta=0.001, window = 5, path='checkpoints/checkpoint.pt', verbose=True):
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
    def __init__(
        self,
        n_fft=4096,
        scale=1,
        power = 2,
        device="cpu"
    ):
        super().__init__()
        
        self.scale = scale
        self.spec = to_device(ta.transforms.Spectrogram(n_fft=n_fft, hop_length=n_fft//4, power=power), device)
        self.amplitude_to_db = ta.transforms.AmplitudeToDB(stype='power')

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample
        # waveform = ta.transforms.resample()

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

        im = transforms.Resize((224, 224))(spec_db)
        
        return im
    
class AudioToMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        fmin=0,
        sample_rate = 16000,
        n_mels=128,
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to power spectrogram
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

        im = transforms.Resize((224, 224))(spec_db)

        return im
    
class AudioToMFCCSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mfcc = 13,
        n_fft=4096,
        n_mels = 23,
        sample_rate =  16000,
        device="cpu"
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

        im = transforms.Resize((224, 224))(spec_db)
        
        return im