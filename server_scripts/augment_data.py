from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import getpass
import numpy as np
import os
import pandas as pd
import paramiko
from pathlib import PurePath
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
import io

thread_local = threading.local()

### SFTP functions ###
def get_sftp():
    """Returns the SFTP client for the current thread, creating it if it doesn't exist."""
    if not hasattr(thread_local, "sftp"):
        time.sleep(random.uniform(0, 256))
        transport = paramiko.Transport((host, port))
        # Use your existing username/password variables here
        transport.connect(username=username, password=password)
        thread_local.transport = transport
        thread_local.sftp = paramiko.SFTPClient.from_transport(transport)
        thread_local.sftp.chdir("./data")
    return thread_local.sftp

def loadPT(input_dir, audio_file):
    """
    Instantiates a torch audio file.

    Input:
        audio_file - path to the .pt-file

    Output: 
        audio_file - the loaded tensor
    """
    sftp = get_sftp()
    with sftp.open(f"{sftp.getcwd()}/{input_dir}/{str(audio_file)}", 'rb') as remote_file:
        file_content = remote_file.read()
        buffer = io.BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))

### Functions for the processing ###
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
    
    # Ensure we do at least 1 iteration if pct > 0 and signal is long enough
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
        sftp = get_sftp()
        print(f"Processing: {file_path}")
        wave = loadPT(input_dir, file_path).to(torch.float16)
        for fun, suf in tasks:
            target_path = output_dir / (f"{suf}_{file_path.stem}.pt")

            # Skip the file already exists
            if target_path.exists():
                print(f"File {suf}_{file_path.stem}.pt already exists.")
                continue
            
            audio_tensor = fun(wave).to(torch.float16)
            print("Processed wave")
            # Save tensor
            buffer = io.BytesIO()
            torch.save(audio_tensor, buffer)

            # Upload the buffer to the SFTP server
            buffer.seek(0)
            with sftp.open(target_path, 'wb') as f:
                f.write(buffer.getbuffer())

            del audio_tensor
        del wave   
        return True
    except Exception as e:
        print(e)
        return False
    

### Execution ###
if __name__ == '__main__':
    host = 'os-login.lsdf.kit.edu'
    port = 22

    transport = paramiko.Transport((host, port))

    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    transport.connect(username = username, password = password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir("./data")

    dawn_dir = PurePath("./AudioTensors_denoised")#_cut")
    xeno_dir = PurePath("./XenoCanto_denoised_cut")
    output_dir = "./Augmented_data_denoised_cut"

    try:
        sftp.mkdir(output_dir)
        print(f"Created directory: {output_dir}")
    except IOError:
        print(f"Directory '{output_dir}' already exists.")

    dawn_files =  [dawn_dir /f for f in sftp.listdir(str(dawn_dir))]
    xeno_files = [xeno_dir /f for f in sftp.listdir(str(xeno_dir))]

    audio_files = [*dawn_files, *xeno_files]


    audio_files_no_rem = [p for p in audio_files if "rem" not in p.stem]

    print(f"Total files to process: {len(audio_files_no_rem)}")

    sftp.close()
    transport.close()

    print("MAX WORKERS: ",os.cpu_count())
    workers = max(1, os.cpu_count())
    
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

    # Remove the audio parts from the audio_files list to only get the id
    cleaned_files = list(map(lambda f: int(str(f).replace("AudioTensors_denoised_cut\\", "").replace("XenoCanto_denoised_cut\\", "").split("_audio")[0].split("_")[2]), audio_files_no_rem))

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
    files_to_process = []

    for file_path in filtered_audio_files:
        exist_counter = 0
        for _, suf in tasks:
            target_path = output_dir / f"{suf}_{file_path.stem}.pt"
            if target_path.exists():
                exist_counter += 1
        
        if exist_counter < 8:
            files_to_process.append(file_path)

    filtered_audio_files = files_to_process
    print(f"Total files to process: {len(filtered_audio_files)}")

    # --- Execution ---
    with ThreadPoolExecutor(max_workers=os.cpu_count()-12) as executor:
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(
            executor.map(lambda f: augment_data_record(f, output_dir), filtered_audio_files), 
            total=len(filtered_audio_files),
            desc="Augmenting data."
        ))


    print(f"Success: {sum(results)} | Failed: {len(results) - sum(results)}")