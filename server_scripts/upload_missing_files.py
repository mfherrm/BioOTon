from concurrent.futures import ProcessPoolExecutor
import getpass
from itertools import repeat
import os 
from pathlib import Path
import paramiko
import random
import time
import tqdm


ssh_storage = {}
def init_worker(host, port, username, password):
    """
    Runs once when each worker process starts.
    Creates a persistent SSH and SFTP connection for this process.
    """
    time.sleep(random.uniform(0, 2))
    transport = paramiko.Transport((host, port))
    transport.connect(username = username, password = password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir("./data")
    print(sftp.getcwd())
    
    # Store them in the global dict to keep them alive
    ssh_storage['sftp'] = sftp

def push_to_lsdf(output_dir, file):
    try:
        sftp = ssh_storage['sftp']
        target_path = f"{output_dir}/{file.name}"
        sftp.put(file, target_path)
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

    input_dir ="./AudioTensors"
    output_dir = "./AudioTensors"

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

    workers = max(1, os.cpu_count() - 2)

    # --- Execution ---
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=init_worker,
                             initargs=(host, port, username, password)
                             ) as executor:
        # Use a list to hold the results so tqdm can track progress
        results = list(tqdm.tqdm(executor.map(push_to_lsdf, repeat(output_dir), files), total=len(files), desc="Uploading data."))

    print(f"Success: {sum(results)} | Failed: {len(results) - sum(results)}")

    sftp.close()
    transport.close()
    print('Finished uploading.')