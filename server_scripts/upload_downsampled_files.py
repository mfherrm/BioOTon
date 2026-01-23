from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import getpass
import io
from functools import partial
from itertools import repeat
import os 
from pathlib import Path
import paramiko
import random
import threading
import time
import torch
import tqdm


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

def loadPT(audio_file):
    """
    Instantiates a torch audio file.

    Input:
        audio_file - path to the .pt-file

    Output: 
        audio_file - the loaded tensor
    """
    return torch.load(audio_file)


def downsample(file_path, output_dir):
    try:
        sftp = get_sftp()
        target_path = f"{sftp.getcwd()}/{output_dir}/{str(file_path.name)}"
        
        print(f"Proccessing: {str(file_path)}")

        # Load wave
        wave = loadPT(file_path).to(torch.float16)

        buffer = io.BytesIO()
        torch.save(wave, buffer)

        # Upload the buffer to the SFTP server
        buffer.seek(0)
        with sftp.open(target_path, 'wb') as f:
            f.write(buffer.getbuffer())
        
        return True
    except Exception as e:
        print(e)


# --- Execution ---
if __name__ == '__main__':
    host = 'os-login.lsdf.kit.edu'
    
    port = 22
    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    transport = paramiko.Transport((host, port))
    transport.connect(username = username, password = password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir("./data")

    input_dir ="./AudioTensors_denoised"
    output_dir = "./AudioTensors_denoised"

    local_files = list(Path(input_dir).glob("*.pt"))
    local_files_list = [file.name for file in local_files]

    remote_files = sftp.listdir(output_dir)

    files = [x for x in local_files_list if x not in remote_files]

    files = [Path(os.getcwd(), output_dir, file) for file in files]


    print(f"Total files to process: {len(files)}")

    try:
        sftp.mkdir(output_dir)
        print(f"Created directory: {output_dir}")
    except IOError:
        print(f"Directory '{output_dir}' already exists.")

    sftp.close()
    transport.close()



    print(f"Total files to process: {len(files)}")

    print("MAX WORKERS: ",os.cpu_count())
    workers = max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        func = partial(downsample, output_dir=output_dir)
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(func, files), total=len(files), desc="Downsampling data."))

    print(f"Success: {sum(results)} | Failed: {len(results) - sum(results)}")


    sftp.close()
    transport.close()
    print('Finished uploading.')



 

    