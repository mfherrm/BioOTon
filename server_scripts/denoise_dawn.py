from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ffmpeg
from functools import partial
import getpass
import io
import os
import paramiko
import random
import time
import torch
import tqdm
import numpy as np
import noisereduce as nr
import threading


thread_local = threading.local()

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
    print(f"Opening from {sftp.getcwd()}/{input_dir}/{str(audio_file)}")
    with sftp.open(f"{sftp.getcwd()}/{input_dir}/{str(audio_file)}", 'rb') as remote_file:
        file_content = remote_file.read()
        buffer = io.BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))

def get_noise_profile(audio_tensor, sr : int, window_duration : float = 1.0):
    """
       Finds the quietest segment to find signal characteristics.

        Input:
            audio_tensor - the signal to be processed
            sr : int - original sampling rate
            window_duration : float - duration of the window to find
        Output:
            tensor - characteristic tensor
    """
    window_samples = int(window_duration * sr)
    # Calculating in strides speeds up the processing
    stride = int(sr * 0.1)

    # If audio is too short, return it all
    if audio_tensor.shape[-1] <= window_samples:
        return audio_tensor

    analysis_wave = audio_tensor[0] if audio_tensor.ndim > 1 else audio_tensor

    # Shape: [num_windows, window_samples]
    windows = analysis_wave.unfold(0, window_samples, stride)

    energies = torch.sqrt(torch.mean(windows**2, dim=1))

    min_idx = torch.argmin(energies).item()
    
    start = min_idx * stride
    end = start + window_samples

    if audio_tensor.ndim > 1:
        return audio_tensor[:, start:end]
    return audio_tensor[start:end]

def denoise_data(file_path, input_dir, output_dir, sampling_rate=16000, window_duration=2.5, target_sr=16000, target_loudness=-16):
    """
       Denoises data by finding characteristic profile, then saves it to the disk.

        Input:
            file_path : str - the directory containing the signals to be processed
            output_dir : str - directory to save the transformed tensors to
            audio_tensor - the signal to be processed
            sampling_rate : int - original sampling rate
            window_duration : float - duration of the window to find
        Output:
            tensor - characteristic tensor
    """
    try:
        sftp = get_sftp()
        target_path = f"{sftp.getcwd()}/{output_dir}/{str(file_path)}_dn.pt"
        
        # Skip the file if it already exists
        # if target_path.exists():
        #     return True
        print(f"Proccessing: {sftp.getcwd()}/{input_dir}/{str(file_path)}")

        # Load wave
        wave = loadPT(input_dir, f"{str(file_path)}")
        
        noise_part = get_noise_profile(wave, sampling_rate, window_duration)


        wave_np = wave.numpy()
        noise_np = noise_part.numpy()

        # Reduce noise
        reduced_noise = nr.reduce_noise(
            y=wave_np, 
            sr=sampling_rate, 
            y_noise=noise_np, 
            n_fft=4096,
            hop_length=204, # Approx 95% overlap (4096 * 0.05)
            prop_decrease=1.0
        )

        # Convert back
        reduced_noise_tensor = torch.from_numpy(reduced_noise).to(torch.float16)

        print("Successfully processed the file")
        buffer = io.BytesIO()
        torch.save(reduced_noise_tensor, buffer)

        # Upload the buffer to the SFTP server
        buffer.seek(0)
        with sftp.open(target_path, 'wb') as f:
            f.write(buffer.getbuffer())
        
        
        return True
    except Exception as e:
        # Returning the error string helps debugging
        print(e)
        return str(e)
    

# --- Execution ---
if __name__ == '__main__':
    host = 'os-login.lsdf.kit.edu'
    port = 22

    transport = paramiko.Transport((host, port))

    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    transport.connect(username = username, password = password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir("./data")

    input_dir ="AudioTensors"
    output_dir = "AudioTensors_denoised"

    files = sftp.listdir(input_dir)

    print(f"Total files to process: {len(files)}")

    try:
        sftp.mkdir(output_dir)
        print(f"Created directory: {output_dir}")
    except IOError:
        print(f"Directory '{output_dir}' already exists or cannot be created.")

    sftp.close()
    transport.close()

    print("MAX WORKERS: ",os.cpu_count())
    workers = max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        func = partial(denoise_data, input_dir = input_dir, output_dir=output_dir, 
                  sampling_rate=16000, window_duration=2.5)
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(func, files), total=len(files), desc="Denoising data."))

    print(f"Success: {sum(results)} | Failed: {len(results) - sum(results)}")

    # with ThreadPoolExecutor(max_workers=workers,
    #                          initializer=init_worker,
    #                          initargs=(host, port, username, password)) as executor:
    #     func = partial(denoise_flac_data, output_dir=output_dir, 
    #                 sampling_rate=16000, window_duration=2.5)

    #     results = list(tqdm.tqdm(
    #         executor.map(func, files), 
    #         total=len(files),
    #         desc="Denoising data"
    #     ))

    # # Check errors
    # failures = [r for r in results if r is not True]
    # print(f"Success: {len(files) - len(failures)} | Failed: {len(failures)}")
    # if failures:
    #     print(f"First error: {failures[0]}")

    sftp.close()
    transport.close()
    print('Finished uploading.')

# if __name__ == '__main__':


#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         func = partial(denoise_flac_data, output_dir=output_dir, 
#                         sampling_rate=16000, window_duration=2.5)

#         results = list(tqdm(
#             executor.map(func, audio_files), 
#             total=len(audio_files),
#             desc="Denoising data"
#         ))

#     # Check errors
#     failures = [r for r in results if r is not True]
#     print(f"Success: {len(audio_files) - len(failures)} | Failed: {len(failures)}")
#     if failures:
#         print(f"First error: {failures[0]}")