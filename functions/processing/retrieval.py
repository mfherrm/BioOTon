from itertools import compress
import geopandas as gpd
import os
import pandas as pd
import numpy as np
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