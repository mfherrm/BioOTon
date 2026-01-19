from concurrent.futures import ThreadPoolExecutor
import getpass
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
import io

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
    with sftp.open(f"{sftp.getcwd()}/{input_dir}/{str(audio_file)}", 'rb') as remote_file:
        file_content = remote_file.read()
        buffer = io.BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))

def cut_file(file, input_dir, output_dir, snippet_length = 15, sample_rate = 16000):
    try:
        sftp = get_sftp()
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

    input_dir ="./AudioTensors_denoised"
    output_dir = "./AudioTensors_denoised_cut"

    files = sftp.listdir(input_dir)

    print(f"Total files to process: {len(files)}")

    try:
        sftp.mkdir(output_dir)
        print(f"Created directory: {output_dir}")
    except IOError:
        print(f"Directory '{output_dir}' already exists.")

    sftp.close()
    transport.close()

    print("MAX WORKERS: ",os.cpu_count())
    workers = max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm.tqdm(
            executor.map(lambda f: cut_file(f, input_dir, output_dir), files), 
                total=len(files),
                desc="Cutting files into 15s parts."
            )
        )

    print(f"Success: {sum(results)} | Failed: {len(results) - sum(results)}")
