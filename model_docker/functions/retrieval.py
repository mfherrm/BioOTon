import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

from functions.sftp import get_pool
#%% Dataset retrieval functions ###
def process_points_dir(file_path, folder, folder_suffix):
    """
    Processes a file path, folder and corresponding suffix and returns recording locations and the corresponding labels. Intended for use in the CombinedSpectroDataset.

    Input:
        file_path - path to a .pt-file containing point labels
        folder - folder NAME where the recordings are located in
        folder_suffix - to enable loading both normal and denoised data 

    Output: 
        file_path - path to the label file
        recordings_path - path to the directory containing the recordings
        point_df - the label dataframe
        audio_files - paths to all audio files
        point_subset - point_df filtered to only include points that are in the file directory
        filtered_audio_files - audio_files filtered to include only files that occur in point_subset
    """
    recordings_path = Path(f"{file_path.parent}/{folder}{folder_suffix}")

    # Read all label points
    point_df = pd.read_parquet(file_path)

    # Get list of all recordings
    audio_files = list(recordings_path.glob("*.pt"))

    # Get the id as an integer
    rec_ids = {int(re.search(r"\d+", f.stem).group()) for f in audio_files}

    # Filter the label frame to only use the audio_files
    point_subset = point_df[point_df['id'].isin(rec_ids)].dropna(subset=['label'])

    # In some cases there are files in the recording directory that do not occur in the data frame
    # Thus, the file directory needs to be filtered to the dataframe as well
    # Map all integer ids to the corresponding file_path
    path_map = {int(re.search(r"\d+", p.stem).group()): p for p in audio_files}

    filtered_audio_files = [path_map[uid] for uid in point_subset['id'] if uid in path_map]


    return file_path, recordings_path, point_df, audio_files, point_subset, filtered_audio_files

def process_cut_points_dir(file_path, folder, folder_suffix, sftp_config: dict, ignore_rem : bool = True):
    """
        Processes a file path, folder and corresponding suffix and returns recording locations and the corresponding labels. Intended for use in the CombinedSpectroDataset.

        Input:
            file_path - path to a .pt-file containing point labels
            folder - folder NAME where the recordings are located in
            folder_suffix - to enable loading both normal and denoised data 

        Output: 
            file_path - path to the label file
            recordings_path - path to the directory containing the recordings
            point_df - the label dataframe
            audio_files - paths to all audio files
            point_subset - point_df filtered to only include points that are in the file directory
            filtered_audio_files - audio_files filtered to include only files that occur in point_subset
    """
    pool = get_pool(sftp_config)
    sftp = pool.get_sftp()
    recordings_path = Path(f"{file_path.parent}/{folder}{folder_suffix}")
    # Read all label points
    with sftp.open(file_path.as_posix(), 'rb') as remote_file:
        print(f"Opened label file {file_path}.")
        point_df = pd.read_parquet(remote_file)
        point_df = point_df[~point_df.label.isna()]
        point_df.id= point_df.id.astype(int)

    # Get list of all recordings
    #audio_files = list(recordings_path.glob("*.pt"))
    audio_files = sftp.listdir(recordings_path.as_posix())
    audio_files = [Path(file) for file in audio_files]

    audio_files_no_rem = [p for p in audio_files if "rem" not in p.stem]

    # Get the id as an integer
    rec_ids = {
        int(re.search(r'\d+', (s := f.name.split("_"))[3 if len(s) >= 6 else 2]).group())
        for f in audio_files_no_rem
    }

    # Filter the label frame to only use the audio_files
    point_subset = point_df[point_df['id'].isin(rec_ids)].dropna(subset=['label'])

    # In some cases there are files in the recording directory that do not occur in the data frame
    # Thus, the file directory needs to be filtered to the dataframe as well
    # Map all integer ids to the corresponding file_path
    path_map = defaultdict(list)
    for p in audio_files_no_rem:
        uid = int(re.search(r'\d+', (s := p.name.split("_"))[3 if len(s) >= 6 else 2]).group())
        path_map[uid].append(p)

    filtered_path_map = {uid:path_map[uid] for uid in point_subset['id'] if uid in path_map}
        
    filtered_audio_files= []
    filtered_audio_labels = []

    for id in filtered_path_map:
        label = int(point_subset[point_subset["id"].isin([id])].label.unique()[0])
        data = filtered_path_map[id]
        cuts = len(data)
 
        for i in range(cuts):
            filtered_audio_files.append(f"{sftp.getcwd()}/{folder}{folder_suffix}/{data[i]}")
            filtered_audio_labels.append(label)


    return file_path, recordings_path, point_df, audio_files, point_subset, filtered_audio_files, filtered_audio_labels