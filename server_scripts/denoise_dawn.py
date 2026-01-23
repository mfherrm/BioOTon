from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ffmpeg
from functools import partial
import getpass
import io
import noisereduce as nr
import numpy as np
import os
import paramiko
from pathlib import Path
import random
import time
import threading
import torch
import torch.nn.functional as F
import tqdm


thread_local = threading.local()
pool = None

class SFTPConnectionPool:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.transport = None
        self._lock = threading.Lock()
        self.thread_local = threading.local()

    def _get_transport(self):
        with self._lock:
            # Check if transport is dead or closed
            if self.transport is None or not self.transport.is_active():
                print(f"[{threading.current_thread().name}] Establishing new Transport...")
                self.transport = paramiko.Transport((self.host, self.port))
                self.transport.connect(username=self.username, password=self.password)
            return self.transport

    def get_sftp(self):
        # If the thread already has a client, check if it's still alive
        if hasattr(self.thread_local, "sftp"):
            try:
                self.thread_local.sftp.listdir('.') # Test the connection
                return self.thread_local.sftp
            except:
                del self.thread_local.sftp # It's dead, remove it

        # Attempt to create a new SFTP client with retries
        for i in range(5): # Try 5 times to get a channel
            try:
                time.sleep(random.uniform(0, 32))
                transport = self._get_transport()
                sftp = paramiko.SFTPClient.from_transport(transport)
                absolute_start_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
                sftp.chdir(absolute_start_path)
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
    
    # Get expected size
    stat = sftp.stat(remote_path)
    expected_size = stat.st_size
    print(expected_size)

    with sftp.open(remote_path, 'rb') as remote_file:
        remote_file.prefetch(expected_size)
        file_content = remote_file.read()
        
        # Verify if the download was complete
        if len(file_content) != expected_size:
            raise IOError(f"Incomplete read: {len(file_content)}/{expected_size} bytes")
            
        buffer = io.BytesIO(file_content)
        try:
            return torch.load(buffer, map_location='cpu')
        except Exception as e:
            print(f"\n[CRITICAL CORRUPTION] File {audio_file} is unreadable: {e}")
            raise e

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
        sftp = pool.get_sftp()
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
    
def denoise_flac_data(file_path, input_dir, output_dir, sampling_rate=16000, window_duration=2.5, target_sr=16000, target_loudness=-16):
    try:
        sftp = pool.get_sftp()
        target_path = f"{sftp.getcwd()}/{output_dir}/{str(file_path.split(".")[0])}_dn.pt"
        
        # Skip the file if it already exists
        # if target_path.exists():
        #     return True
        print(f"Proccessing: {sftp.getcwd()}/{input_dir}/{str(file_path)}")
        
        # Load flac file and process it 
        with sftp.open(f"{sftp.getcwd()}/{input_dir}/{str(file_path)}", 'rb') as remote_in:
            input_bytes = remote_in.read()
            
            out, _ = (
                ffmpeg
                .input('pipe:')
                # Normalize the Loudness
                .filter('loudnorm', i=target_loudness, tp=-1.0, lra=11)
                # ac=1: Convert to mono, ar: sample at target sr, 'f32le': output 32-bit float Little Endian
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=target_sr)
                .run(input=input_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
            )

            # Convert bytes to numpy array
            audio_np = np.frombuffer(out, np.float32)

            # Convert to 16-bit Float tensor
            wave = torch.from_numpy(audio_np.copy()).to(torch.float16)
            
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
        print(f"Error: {sftp.getcwd()}/{input_dir}/{str(file_path)} -> {str(e)}")
        return str(e)

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
    input_dir ="Dawn_chorus_conversion_flac"
    output_dir = "Dawn_denoised"

    files = main_sftp.listdir(input_dir)
    files = [file for file in files if file.endswith(".flac")]
    print(files)

    print(f"Total files to process: {len(files)}")

    try:
        main_sftp.mkdir(output_dir)
    except IOError:
        print(f"Directory '{output_dir}' already exists.")
        pass

    print("MAX WORKERS: ",os.cpu_count())
    workers = 16 #max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # func = partial(denoise_data, input_dir = input_dir, output_dir=output_dir, 
            # sampling_rate=16000, window_duration=2.5)
        func = partial(denoise_flac_data, input_dir = input_dir, output_dir=output_dir, 
            sampling_rate=16000, window_duration=2.5)
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(func, files), total=len(files), desc="Denoising data."))

    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    print(f"Success: {success_count} | Failed: {fail_count}")
    
    print('Finished uploading.')