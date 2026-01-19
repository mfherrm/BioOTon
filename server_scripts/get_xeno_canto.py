import getpass
import numpy as np
import os
import pandas as pd
import paramiko
from pathlib import Path
import requests

host = 'os-login.lsdf.kit.edu'
# host = "os-webdav.lsdf.kit.edu"
port = 22
username = input("Enter username: ") or "uyrra"
password = getpass.getpass("Enter password: ")

transport = paramiko.Transport((host, port))
transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)
sftp.chdir("./data")
print(sftp.getcwd())

# Set the base directory to your target folder
base_directory = r'./XenoCanto'
try:
    sftp.mkdir(base_directory)
    print(f"Created directory: {base_directory}")
except IOError:
    print(f"Directory '{base_directory}' already exists or cannot be created.")

# Get number of pages with the results
api_url = f"https://xeno-canto.org/api/3/recordings?query=cnt:germany+len:30-60+grp:birds&key=7894b51f8bcdf05cfac66a1455cbb314c4313486"
print(f"→ Querying {api_url}")
response = requests.get(api_url)
if response.status_code != 200:
    print(f"API request failed.")

# process response to get page numbers
data = response.json()
page_number = data["numPages"]

recordings_list = []

# Build frame with all recording ids and lat / lon
for p in np.arange(1, page_number+1):
    api_url = f"https://xeno-canto.org/api/3/recordings?query=cnt:germany+len:30-60+grp:birds&key=7894b51f8bcdf05cfac66a1455cbb314c4313486&page={p}"
    print(f"→ Querying {api_url}")
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"API request failed.")

    data = response.json()
    recordings = data.get('recordings', [])

    
    if not recordings:
        print(f"No recordings found.")

    recordings = pd.DataFrame.from_dict(recordings)

    recordings_list.append(recordings)

recordings_frame = pd.concat(recordings_list, ignore_index=True)
recordings_frame = recordings_frame.rename(columns={"lon":"lng"})


total_downloaded = 0
for recording_id in recordings_frame.id:
    try:
        # Download URL using the recording ID
        file_url = f"https://xeno-canto.org/{recording_id}/download"
        filename = f"{recording_id}_audio.flac"
        save_path = f"{base_directory}/{filename}"
        
        with requests.get(file_url, stream=True) as r:
            if r.status_code == 200:
                with sftp.open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename}")
                total_downloaded += 1
            else:
                print(f"Failed to download {filename} (status: {r.status_code})")
    except Exception as e:
        print(f"Error downloading recording {recording_id}: {e}")

sftp.close()
transport.close()
print('Finished fetching Xeno-Canto data.')