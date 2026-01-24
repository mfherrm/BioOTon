from concurrent.futures import ThreadPoolExecutor
import ffmpeg
from functools import partial
import geopandas as gpd
import getpass
import io
import noisereduce as nr
import numpy as np
import os
import pandas as pd
import paramiko
from pathlib import Path, PurePath
import random
import re
import threading
import time
from torchaudio.functional import add_noise
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
import tqdm


thread_local = threading.local()
pool = None

class SFTPConnectionPool:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.thread_local = threading.local()

    def get_sftp(self):
        # If the thread already has a client, check if it's still alive
        if hasattr(self.thread_local, "sftp"):
            try:
                if self.thread_local.sftp.get_channel().get_transport().is_active():
                    return self.thread_local.sftp
            except:
                pass

        # Attempt to create a new SFTP client with retries
        for i in range(5): # Try 5 times to get a channel
            try:
                time.sleep(random.uniform(0, 16))
                # transport = self._get_transport()
                # sftp = paramiko.SFTPClient.from_transport(transport)
                transport = paramiko.Transport((self.host, self.port))
                transport.connect(username=self.username, password=self.password)
                sftp = paramiko.SFTPClient.from_transport(transport)
                absolute_start_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
                try:
                    sftp.chdir(absolute_start_path)
                except IOError:
                    print(f"Warning: Could not chdir to {absolute_start_path}")
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

def loadPT(input_dir, audio_file):
    """
    Instantiates a torch audio file.

    Input:
        audio_file - path to the .pt-file

    Output: 
        audio_file - the loaded tensor
    """
    sftp = pool.get_sftp()
    remote_path = f"{sftp.getcwd()}/{input_dir}/{str(audio_file)}"
    print("Processing: ", remote_path)
    

    with sftp.open(remote_path, 'rb') as remote_file:
        remote_file.prefetch(1024 * 384)
        file_content = remote_file.read()
        buffer = io.BytesIO(file_content)
        try:
            return torch.load(buffer, map_location='cpu')
        except Exception as e:
            print(f"\n[CRITICAL CORRUPTION] File {audio_file} is unreadable: {e}")
            raise e
        
# Augmentation functions
def selectSubset(frame, wanted_classes):
    """
    Selects a subset of a dataframe according to given class labels.

    Input:
        frame - the dataframe to be transformed, must include a "label" column
        wanted_classes - the classes to select

    Output: 
        frame - a dataframe containing only the rows that have the given class labels
    """
    wanted_indices = []
    for cl in wanted_classes:
        indices = []
        for idx, st in enumerate(frame["label"]):
            indices.append(str(st).startswith(cl))
        wanted_indices.append(indices)

    return frame.loc[np.any(wanted_indices, axis=0)]

def add_white_noise(signal, snr = 30):
    """
        snr: in db, use ~ 30 for low noise
    """
    signal = signal.unsqueeze(0)
    noise = torch.randn_like(signal)

    noisy_signal = add_noise(signal, noise, torch.tensor([snr]))
    
    return noisy_signal

def random_volume_change(signal, sampling_rate : int = 16000, pct : float = 0.1, seconds : float = 1.0):
    """
        Increases / decreases the volume of random parts of a signal.

        Input:
            signal - the signal to be processed
            sampling_rate : int - sampling rate of the signal
            pct : float - percentage of the signal to be affected
            seconds : float - duration of affected change per occurence

        Output:
            tensor - tensor with changed volume
    """
    clone = signal.clone()
    sig_len = signal.shape[-1]
    
    window_size = int(sampling_rate * seconds)
    
    num_iterations = int((sig_len * pct) / window_size)
    
    # Ensure signal is long enough
    num_iterations = max(num_iterations, 1) if pct > 0 else 0

    for i in range(num_iterations):

        if sig_len > window_size:
            cut_idx = np.random.randint(0, sig_len - window_size)

            clone[..., cut_idx : cut_idx + window_size] *= 7
            
    return clone


def random_cutout(signal, pct :float = .15, **kwargs):
    """
        Adds random grain to the signal.

        Input:
            signal - the signal to be processed
            pct : percentage of the signal to be made grainy

        Output:
            tensor - tensor with grain effect
    """
    copy = signal.clone()
    sig_len = signal.shape[0]
    sigs_to_cut = int(sig_len * pct)
    for i in range(0, sigs_to_cut):
        cut_idx = random.randint(0, sig_len - 1)
        copy[cut_idx] = 0
    return copy


def pitch_warp(signal, sr : int = 16000, sr_divisor : float = 2.0, **kwargs):
    """
       Performs lossy sampling on a signal.

        Input:
            signal - the signal to be processed
            sr : int - sampling rate of the original signal
            sr_divisor - proportion of the sampling

        Output:
            tensor - lossily sampled tensor
    """
    down_sr = sr // sr_divisor
    resample_down = T.Resample(orig_freq=sr, new_freq=down_sr).to(signal.device, signal.dtype)
    resample_up = T.Resample(orig_freq=down_sr, new_freq=sr).to(signal.device, signal.dtype)
    return resample_up(resample_down(signal))


def random_timeshift(signal):
    """
        Cuts up a signal at random places and rearranges them.

        Input:
            signal - the signal to be processed

        Output:
            tensor - tensor with changed signal places
    """
    max = len(signal)
    borders = np.sort(np.random.rand(2))

    floor = np.floor(max*borders[0]).astype(int)
    ceil = np.floor(max*borders[1]).astype(int)

    shifted = torch.cat([signal[floor:ceil], signal[:floor], signal[ceil:]])

    return shifted


def vertical_blackout(signal, pct : float = 0.1):
    """
        Blacks out the vertical portion of the spectrogram by applying a transformation to a signal.

        Input:
            signal - the signal to be processed
            pct : float - percentage of the signal to be blacked out

        Output:
            tensor - tensor with vertical blackout
    """
    m = len(signal)
    border = np.sort(np.random.rand(1))[0]
    
    floor = np.floor(m*border).astype(int)
    ceil = np.floor(m*(border+pct)).astype(int)

    csignal = signal.clone()
    csignal[floor:ceil] = 0

    return csignal


def horizontal_blackout(signal, sample_rate, center_freq, pct=.05):
    """
        Blacks out the vertical portion of the spectrogram by applying a transformation to a signal.

        Input:
            signal - the signal to be processed
            sample_rate - sample rate of the signal
            center_freq - frequency for the bandpass filter
            pct : float - part of the signal to be blacked out

        Output:
            tensor - tensor with horizontal blackout
    """

    center_freq = np.random.uniform(0.15, 1.0) * center_freq
    
    nyquist = sample_rate / 2
    bandwidth = nyquist * pct
    
    Q = center_freq / max(bandwidth, 1e-6)

    print(Q)
    
    band = AF.bandpass_biquad(signal, sample_rate, center_freq, Q=Q)
        
    return signal - band


def cut_off_edge(signal):
    """
        Cuts the left / right edge off a signal.

        Input:
            signal - the signal to be processed

        Output:
            tensor - tensor with cut off edge
    """
    coin_flip = np.random.randint(2)

    clone = signal.clone()
    offset = 16000 * 10

    if coin_flip == 0:
        clone[..., :-offset] = signal[..., offset:].clone()
    else: 
        clone[..., offset:] = signal[..., :-offset].clone()

    return clone


# Main processing function
def augment_data_record(file_path, input_dir, output_dir):
    tasks = [
        (lambda val: add_white_noise(val, snr = 60 ), "wn"),
        # (lambda val: speed_up(val, factor = 1.15), "sp"),
        (lambda val: random_volume_change(val, pct=.15), "vc"),
        (lambda val: random_cutout(val, pct=.00025), "rc"),
        (lambda val: pitch_warp(val, sr_divisor=1.15), "pw"),
        (lambda val: random_timeshift(val), "rt"),
        # (lambda val: oversample(val, new_freq=20000), "os"),
        (lambda val: vertical_blackout(val), "vb"),
        (lambda val: horizontal_blackout(val.float(), 16000, 8000, pct=0.5), "hb"),
        (lambda val: cut_off_edge(val), "ce")
    ]
    try:
        sftp = pool.get_sftp()
        print(f"Processing: {file_path}")
        wave = loadPT(input_dir, file_path).to(torch.float16)
        for fun, suf in tasks:
            target_path = output_dir / (f"{suf}_{file_path.stem}.pt")

            # # Skip the file already exists
            # if target_path.exists():
            #     print(f"File {suf}_{file_path.stem}.pt already exists.")
            #     continue
            
            audio_tensor = fun(wave).to(torch.float16)
            print("Processed wave")
            # Save tensor
            buffer = io.BytesIO()
            torch.save(audio_tensor, buffer)

            # Upload the buffer to the SFTP server
            buffer.seek(0)
            with sftp.open(str(target_path), 'wb', bufsize=32768) as f:
                f.write(buffer.getbuffer())

            del audio_tensor
        del wave   
        return True
    except Exception as e:
        print(f"Error: {sftp.getcwd()}/{input_dir}/{str(file_path)} -> {str(e)}")
        return False



# --- Execution ---
if __name__ == '__main__':
    host = 'os-login.lsdf.kit.edu'
    port = 22

    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    pool = SFTPConnectionPool(host, port, username , password)

    main_sftp = pool.get_sftp()
    # main_sftp.chdir(".")
    print(main_sftp.getcwd())
    # main_sftp.chdir("../../../ipf/projects/Bio-O-Ton/Audio_data")

    output_dir = "Augmented_data_denoised_cut"

    try:
        main_sftp.mkdir(output_dir)
    except IOError:
        print(f"Directory '{output_dir}' already exists.")
        pass

    dawn_dir = PurePath("Dawn_denoised_cut")
    xeno_dir = PurePath("XenoCanto_denoised_cut")

    dawn_files =  [dawn_dir /f for f in main_sftp.listdir(str(dawn_dir))]
    xeno_files = [xeno_dir /f for f in main_sftp.listdir(str(xeno_dir))]

    audio_files = [*dawn_files, *xeno_files]


    audio_files_no_rem = [p for p in audio_files if "rem" not in p.stem]

    print(f"Original files to process: {len(audio_files_no_rem)}")

    # Filter to only use classes that do not have as much data
    wanted_oc = [
        # Closed off areas
        #  "112", #discontinuous urban fabric # Not used here since it already has roughly 50k points # subset.groupby("label").count()
        "221","222", "244", # Vineyards, Fruit tree and berry plantations (2/3 are trees), Agro-forestry
        "31", # Forest

        # Open areas
        "141", # green urban areas
        "211", "212", "23", "241", "242", "243", # arable land (non- / permanently irrigated), pastures, annual crops, complex cultivation, agriculture+natural vegetation 
        "32", # Shrub and herbaceous
        "41" # Inland wetlands
    ]

    # open label table for dawn chorus
    pframe_single = pd.read_parquet("points_single.parquet")
    pframe_single["geometry"] = gpd.GeoSeries.from_wkb(pframe_single.geometry).buffer(200, cap_style='square')

    # open label table for xeno-canto
    xeno_frame_single = pd.read_parquet("xeno_points_single.parquet")
    xeno_frame_single["geometry"] = gpd.GeoSeries.from_wkb(xeno_frame_single.geometry).buffer(200, cap_style='square')

    # select subsets that contain only the labels above
    dawn_subset = selectSubset(pframe_single, wanted_oc)
    xeno_subset = selectSubset(xeno_frame_single, wanted_oc)
    xeno_subset['id'] = xeno_subset['id'].astype(int)

    print(audio_files_no_rem[0])
    # Remove the audio parts from the audio_files list to only get the id
    cleaned_files = list(map(lambda f: int(str(f).replace("Dawn_denoised_cut/", "").replace("Dawn_denoised_cut\\", "").replace("XenoCanto_denoised_cut/", "").replace("XenoCanto_denoised_cut\\", "").replace("\\", "/").split("_audio")[0].split("_")[2]), audio_files_no_rem))

    # Match the subsets and the audio file ids
    dawn_df = dawn_subset[dawn_subset.id.isin(cleaned_files)]
    xeno_df = xeno_subset[xeno_subset.id.isin(cleaned_files)]

    # combine to get a singular list
    subset = pd.concat([dawn_df, xeno_df])

    id_set = set(subset['id'].astype(str))

    # Match the list with the audio files to get only the paths of files that are relevant
    # this also matches the 10 recordings with more to their name, e.g., August-XX-XX-XX
    # Need to split "_" and select second entry for cut files and their digit so to make the regex not stuck on the digit denoting the cut number 
    filtered_audio_files = [
        p for p in audio_files_no_rem
        if (match := re.search(r'\d+', p.name.split("_")[2])) and match.group(0) in id_set
    ]

    tasks = ["wn", "vc", "rc", "pw", "rt", "vb", "hb", "ce"]
    files_to_process = []

    existing_files = main_sftp.listdir(f"{output_dir}")
    for file_path in filtered_audio_files:
        missing_augmentations = False
        for suf in tasks:
            if f"{suf}_{file_path.stem}.pt" not in existing_files:
                missing_augmentations = True
                break
        if missing_augmentations:
            files_to_process.append(file_path)

    filtered_audio_files = files_to_process
    print(f"Total files to process: {len(filtered_audio_files)}")


    print("MAX WORKERS: ",os.cpu_count())
    workers = 16 #max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # func = partial(denoise_data, input_dir = input_dir, output_dir=output_dir, 
            # sampling_rate=16000, window_duration=2.5)
        # func = partial(augment_data_record, input_dir = filtered_audio_files, output_dir=output_dir)
        func = partial(augment_data_record, input_dir="", output_dir=Path(output_dir))
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(func, filtered_audio_files), total=len(filtered_audio_files), desc="Augmenting data records."))

    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    print(f"Success: {success_count} | Failed: {fail_count}")
    
    print('Finished uploading.')