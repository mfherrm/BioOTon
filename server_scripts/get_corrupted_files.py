from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import getpass
import os
import paramiko
from pathlib import Path
import random
import time
import threading
import torch
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
                sftp.chdir("./data")
                self.thread_local.sftp = sftp
                return sftp
            except Exception as e:
                wait = (i + 1) * 2
                print(f"Channel failed, retrying in {wait}s... Error: {e}")
                time.sleep(wait)
                # Force transport reset on 3rd failure
                if i == 2: self.transport = None 
        
        raise RuntimeError("Could not connect to SFTP after 5 attempts")
    

def replace_file(file_name, local_input_dir, remote_output_dir):
    try:
        sftp = pool.get_sftp()
        local_path = os.path.join(local_input_dir, file_name)
        
        if not os.path.exists(local_path):
            return f"Local Skip: {file_name} not found."

        # Verify local file isn't empty (0 bytes)
        if os.path.getsize(local_path) == 0:
            return f"Local Skip: {file_name} is 0 bytes."

        # Load Local Tensor
        tensor = torch.load(local_path, map_location='cpu', weights_only=False).to(torch.float16)

        # Prepare Remote Path
        # We ensure no leading dots or weird slashes
        clean_remote_dir = remote_output_dir.strip("./")
        remote_out_path = f"{clean_remote_dir}/{Path(file_name).name}"

        # UPLOAD LOGIC
        buf = io.BytesIO()
        torch.save(tensor, buf)
        file_size = buf.tell() # Get size of the buffer
        buf.seek(0)
        
        try:
            # Check if directory exists/is writable by trying to 'stat' it
            sftp.stat(clean_remote_dir)
            
            with sftp.open(remote_out_path, 'wb') as f:
                # Set a standard file attribute (optional, helps some servers)
                f.set_pipelined(True) 
                f.write(buf.getbuffer())
            return True
            
        except IOError as e:
            # Catch specific SFTP Failure (usually code 4)
            return f"SFTP Write Error (Check Quota/Permissions): {str(e)}"

    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"Detailed Error for {file_name}: {error_detail}")
        return f"{type(e).__name__}: {str(e)}"


# --- Execution ---
if __name__ == '__main__':
    host = 'os-login.lsdf.kit.edu'
    port = 22

    username = input("Enter username: ") or "uyrra"
    password = getpass.getpass("Enter password: ")

    pool = SFTPConnectionPool(host, port, username, password)

    main_sftp = pool.get_sftp()
    main_sftp.chdir(".")

    input_dir = f"./AudioTensors_denoised"
    output_dir = f"./AudioTensors_denoised_cut"

    files = main_sftp.listdir(output_dir)

    processed_files = [f for f in files if f.startswith("cut_0")]
    cut_processed_files = [x[6:] for x in processed_files]

    print(f"Found {len(cut_processed_files)} files that are not corrupted.")

    local_files = list(Path(input_dir).glob("*.pt"))
    local_files_list = [file.name for file in local_files]


    corrupted_files = [x for x in local_files_list if x not in cut_processed_files]

    print(f"Total files to process: {len(corrupted_files)}")

    try:
        main_sftp.mkdir(output_dir)
    except IOError:
        print(f"Directory '{output_dir}' already exists.")
        pass

    print("MAX WORKERS: ",os.cpu_count())
    workers = 4 #max(1, os.cpu_count())
    
    # --- Execution ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm.tqdm(
            executor.map(lambda f: replace_file(f, input_dir, output_dir), corrupted_files), 
                total=len(corrupted_files),
                desc="Replacing files."
            )
        )

    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    print(f"Success: {success_count} | Failed: {fail_count}")