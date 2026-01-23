import getpass
import io
import pandas as pd
import paramiko
from pathlib import Path


host = 'os-login.lsdf.kit.edu'

port = 22
username = input("Enter username: ") or "uyrra"
password = getpass.getpass("Enter password: ")

transport = paramiko.Transport((host, port))
transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)

absolute_start_path = "/lsdf01/lsdf/kit/ipf/projects/Bio-O-Ton/Audio_data"
sftp.chdir(absolute_start_path)


augmented_dir = Path(f"{sftp.getcwd()}/augmented_data_denoised")
xeno_points_dir = Path(f"{sftp.getcwd()}/xeno_points_single.parquet")
dawn_points_dir = Path(f"{sftp.getcwd()}/points_single.parquet")

with sftp.open(dawn_points_dir.as_posix(), 'rb') as remote_file:
    dawn_points = pd.read_parquet(remote_file)
with sftp.open(xeno_points_dir.as_posix(), 'rb') as remote_file:
    xeno_points = pd.read_parquet(remote_file)
    xeno_points.id = xeno_points.id.astype(int)

augmented_files = list(augmented_dir.glob("*.pt"))

augmented_ids = [int(str(f).split("_")[-3]) for f in augmented_files]
augmented_points = pd.DataFrame({"id":augmented_ids})

adf = augmented_points.merge(xeno_points, how="left", left_on="id", right_on="id")

bdf = adf.merge(dawn_points, how="left", left_on="id", right_on="id")

bdf["label"] = bdf['label_x'].combine_first(bdf['label_y'])
bdf["geometry"] = bdf['geometry_x'].combine_first(bdf['geometry_y'])

bdf.drop(columns=["label_x", "label_y", "geometry_x", "geometry_y"], inplace=True)

buffer = io.BytesIO()
bdf.to_parquet(buffer)

# Upload the buffer to the SFTP server
buffer.seek(0)
with sftp.open("augmented_points_single.parquet", 'wb') as f:
    f.write(buffer.getbuffer())
# bdf.to_parquet("augmented_points_single.parquet")