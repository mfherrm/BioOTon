import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchaudio as ta

from functions.dl.convenience_functions import to_device

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


class Squeeze_Excite(nn.Module):
    
    def __init__(self,channel,reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class VGGBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels,8)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)
        
        return(out)
    

# https://www.kaggle.com/code/bestofbests9/unet-unet-doubleunet-implementation-pytorch
class NestedUNet(nn.Module):
    
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _up_and_concat(self, skip_tensors, x_to_upsample):
        """
        skip_tensors: A list of tensors from previous stages
        x_to_upsample: The tensor coming from the deeper layer
        """
        x_up = self.up(x_to_upsample)
        
        # We use the first skip tensor as the size template
        target_h = skip_tensors[0].size(2)
        target_w = skip_tensors[0].size(3)
        
        # Calculate padding needed for x_up
        diff_h = target_h - x_up.size(2)
        diff_w = target_w - x_up.size(3)
        
        # pad: (left, right, top, bottom)
        x_up = F.pad(x_up, [diff_w // 2, diff_w - diff_w // 2,
                            diff_h // 2, diff_h - diff_h // 2])
        
        return torch.cat([*skip_tensors, x_up], dim=1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.conv0_1(self._up_and_concat([x0_0], x1_0))

        x2_0 = self.conv2_0(self.pool(x1_0))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_1 = self.conv1_1(self._up_and_concat([x1_0], x2_0))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2(self._up_and_concat([x0_0, x0_1], x1_1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        # x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_1 = self.conv2_1(self._up_and_concat([x2_0], x3_0))
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x1_2 = self.conv1_2(self._up_and_concat([x1_0, x1_1], x2_1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.conv0_3(self._up_and_concat([x0_0, x0_1, x0_2], x1_2))

        x4_0 = self.conv4_0(self.pool(x3_0))
        # x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x3_1 = self.conv3_1(self._up_and_concat([x3_0], x4_0))
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x2_2 = self.conv2_2(self._up_and_concat([x2_0, x2_1], x3_1))
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x1_3 = self.conv1_3(self._up_and_concat([x1_0, x1_1, x1_2], x2_2))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.conv0_4(self._up_and_concat([x0_0, x0_1, x0_2, x0_3], x1_3))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output