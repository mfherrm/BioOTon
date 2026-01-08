from itertools import compress
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import torch


"""

"""
def getSoundLocations(dir):
    dir_files = os.listdir(dir)
    soundscape_file = os.path.join(dir, list(compress(dir_files, [file.endswith(".parquet") for file in dir_files]))[0])
    sound_df = pd.read_parquet(soundscape_file)

    sound_df = sound_df[["id", "lat", "lng"]]

    sound_gdf = gpd.GeoDataFrame(
        sound_df, geometry=gpd.points_from_xy(sound_df.lng, sound_df.lat), crs="EPSG:4326"
    )

    sound_gdf.to_crs("EPSG:3035", inplace=True)

    sound_gdf.drop(["lat", "lng"], axis=1,inplace=True)

    return sound_gdf

def selectSubset(frame, wanted_classes):
    wanted_indices = []
    for cl in wanted_classes:
        indices = []
        for idx, st in enumerate(frame["label"]):
            indices.append(str(st).startswith(cl))
        wanted_indices.append(indices)

    return frame.loc[np.any(wanted_indices, axis=0)]

def transformSubset(frame, codes):
    frame.loc[frame['label'].isin(codes), 'label'] = '0'
    frame.loc[frame['label'] != '0', 'label'] = '1'
    return frame

def loadPT(audio_file):
    return torch.load(audio_file)

def process_points_dir(file_path, folder, folder_suffix):
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