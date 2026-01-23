from concurrent.futures import ThreadPoolExecutor
from functools import partial
import ffmpeg
import getpass
import io
import noisereduce as nr
import numpy as np
import os
import paramiko
from pathlib import Path
import random
import threading
import time
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


def cut_file(file, input_dir, output_dir, snippet_length = 15, sample_rate = 16000):
    try:
        sftp = pool.get_sftp()
        hz_length = snippet_length*sample_rate

        tensor = loadPT(input_dir, file).to(torch.float16)

        tensor = tensor[~tensor.isnan()]

        tensor_length = tensor.shape[-1]
    
        snippets = int(tensor_length / hz_length)
        remainder = tensor_length % hz_length

        rem = False

        min_val = tensor[~tensor.isnan()].min()
        
        if  remainder > 0.85:
            rest_length = hz_length - remainder
            tensor = F.pad(tensor, (0, rest_length), value=min_val)
            snippets +=1
            rem = True

        span = np.arange(snippets)

            
        for snippet in span:
            if rem and snippet == span[-1]:
                tensor_snippet_name = f"{output_dir}/cut_{snippet}_rem_{Path(file).stem}.pt"
            else:
                tensor_snippet_name = f"{output_dir}/cut_{snippet}_{Path(file).stem}.pt"
            
            # Skip the file if it already exists
            # if tensor_snippet_name.exists():
            #     print(f"File {tensor_snippet_name} already exists, skipping...")
            #     continue

            tensor_snippet = tensor[hz_length*snippet:hz_length*(snippet+1)].to(torch.float16)

            buffer = io.BytesIO()
            torch.save(tensor_snippet, buffer)

            # Upload the buffer to the SFTP server
            buffer.seek(0)
            with sftp.open(tensor_snippet_name, 'wb') as f:
                f.write(buffer.getbuffer())

        return True
    except Exception as e:
        # Returning the error string helps debugging
        print(f"Error: {sftp.getcwd()}/{input_dir}/{str(file)} -> {str(e)}")
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
    input_dir = "Dawn_denoised"
    output_dir = "Dawn_denoised_cut"

    files = main_sftp.listdir(input_dir)
    files = [file for file in files if file.endswith(".pt")]
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
        func = partial(cut_file, input_dir = input_dir, output_dir=output_dir)
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(func, files), total=len(files), desc="Cutting files into 15s parts."))

    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    print(f"Success: {success_count} | Failed: {fail_count}")
    
    print('Finished uploading.')