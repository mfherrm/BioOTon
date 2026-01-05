from dbfread import DBF
import ffmpeg
import geopandas as gpd
from itertools import compress
import noisereduce as nr
import numpy as np
import os
import pandas as pd
import random
import rasterio
from rasterio.mask import mask

import torch
import torchaudio.transforms as T

from functions.processing.retrieval import getSoundLocations
from functions.processing.retrieval import loadPT

"""
    Processes recordings according to the range parameter x_range.

"""
def processSingleRecording(corine_dir, recording_dir, distance = 200, x_range='all', mode="simple"):
    dir_files = os.listdir(corine_dir)

    raster_file = os.path.join(corine_dir, list(compress(dir_files, [file.endswith("clip.tif") for file in dir_files]))[0])
    dbf_file = os.path.join(corine_dir, list(compress(dir_files, [file.endswith("vat.dbf") for file in dir_files]))[0])

    # Get DBF File
    dbf_data = DBF(dbf_file)
    dbf_df = pd.DataFrame(iter(dbf_data))

    points = getSoundLocations(recording_dir)
    points["label"] = np.nan
    points["close"] = np.nan

    print("Loading raster.")

    if type(x_range) == str:
        x_range = np.arange(len(points))

    with rasterio.open(raster_file) as tif:
        print("Loaded raster successfully. ")
        raster_crs = tif.crs

        # Reproject points
        if points.crs != raster_crs:
            print(f"Reprojecting points due to differing SRS.")
            points = points.to_crs(raster_crs)

        # Get raster resolution to compute pixel size
        x_res, __ = tif.res

        # Needed for plotting
        all_geo_frames = []
        all_filtered_frames = []
        all_grouped_frames = []
        all_weighted_frames = []

        for idx, content in points.iloc[x_range].iterrows():

            geo_dfs = []

            buffers = [gpd.GeoSeries(content.geometry, crs=raster_crs).buffer(d) for d in np.arange(start=distance/3, step=distance/3, stop= distance+1)]

            print("Generated buffers.")
            # print(buffers)

            image_data, transformed_data = mask(
                    dataset=tif,
                    shapes=buffers[2].to_crs(raster_crs).geometry.tolist(),
                    crop=True,
                    all_touched=True 
                )
            
            print(f"Clipped raster to the largest buffer. \nGenerating individual pixels.")

            # Get shape
            rows, cols = image_data.shape[1:]
            # print(rows, cols)
            # Flatten image to a single band
            pixel_values = image_data[0].flatten()

            # Create arrays of all row and column indices
            col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))

            xs, ys = rasterio.transform.xy(transformed_data, row_indices.flatten(), col_indices.flatten())

            # print(xs, ys)

            geometries = gpd.points_from_xy(xs, ys)#.buffer(x_res/ 2, cap_style='square')[0]]

            # Create geodataframe
            geo_df = gpd.GeoDataFrame(
                pd.DataFrame({
                    'value': pixel_values,
                }),
                geometry=geometries,
                crs=raster_crs
            )

            # If squares are not explicitly needed, comment this out
            geo_df['geometry'] = geo_df.buffer(x_res / 2, cap_style="square")

            # print(geo_df)

            all_geo_frames.append(geo_df)

            for i, buffer in enumerate(buffers):
                if i > 0:
                    buffer = buffers[i].difference(buffers[i-1])

                results = geo_df.intersects(buffer.geometry.iloc[0], align=True)#.clip(buffer)

                # print(results)

                print(f"Selected intersecting pixels for buffer {i}.")

                geo_dfs.append(geo_df[results])
                

            # drop pixels that occur in multiple buffers
            first_ring = pd.concat([geo_dfs[0], geo_dfs[1]])
            second_ring = pd.concat(geo_dfs)

            geo_dfs[1] = first_ring.loc[first_ring.normalize().drop_duplicates(keep=False).index]
            geo_dfs[2] = second_ring.loc[second_ring.normalize().drop_duplicates(keep=False).index]

            print("Dropped duplicate pixels")

            filtered_dfs = []

            for pos, frame in enumerate(geo_dfs):
                # Get only intersecting pixels
                # Need to combine the three buffers into one large one
                intersection_result = frame.intersects(pd.concat(buffers).union_all())

                selection = frame[intersection_result]
                joined_df = selection.merge(dbf_df, left_on = "value", right_on="Value")
            
                filtered_dfs.append(joined_df)

            all_filtered_frames.append(filtered_dfs)
            
            grouped_dfs = []

            for pos, frame in enumerate(filtered_dfs):
                grouped_frame = filtered_dfs[pos][["CODE_18", "geometry"]].groupby("CODE_18").count()
                print(grouped_frame)
                if grouped_frame.empty:
                    print(f"gdf empty")
                    grouped_frame = pd.DataFrame({'CODE_18':[-128], 'geometry': [0]})
                    grouped_frame.set_index('CODE_18', inplace=True)
                    
                grouped_dfs.append(grouped_frame)

            all_grouped_frames.append(grouped_dfs)

            # print(grouped_dfs)
            # Weigh classes and find class with highest weight
            if mode == 'simple':
                weighted_frame = (5 * grouped_dfs[0]).add((3 * grouped_dfs[1]), fill_value=0).add((1 * grouped_dfs[2]), fill_value=0)
            # Inverse squared distance weighting
            elif mode == 'isdw':
                weights = np.arange(start=distance/3, step=distance/3, stop= distance+1)
                squared_weights = weights**2
                weighted_frame = (5000/squared_weights[0] * grouped_dfs[0]).add((5000/squared_weights[1] * grouped_dfs[1]), fill_value=0).add((5000/squared_weights[2] * grouped_dfs[2]), fill_value=0)
            
            print("Weighted classes")#:", weighted_frame)

            # Ignore NAN
            weighted_frame.loc[weighted_frame.index==-128, 'geometry'] = 0
            all_weighted_frames.append(weighted_frame)
            print(weighted_frame)

            # How to handle identical values??, e.g. recording 402
            weighted_class = weighted_frame.idxmax().item()

            print(f"Assigned class {weighted_class} to point {idx}")

            try:
                first, second = weighted_frame.geometry.drop_duplicates().nlargest(2).values
                ratio = first/second
                # There should never be case where a label is selected and another is larger
                # There might be cases where two labels have the same score
                if (ratio > 0.99) & (ratio <  1.15):
                    close = True
                else:
                    close = False
            except:
                print("CLOSE FAILED")
                close = False

            

            points.loc[points.index==idx, 'label'] = weighted_class

            points.loc[points.index==idx, 'close'] = close

    return points, dbf_df, all_geo_frames, all_filtered_frames, all_grouped_frames, all_weighted_frames, raster_crs


"""
    Processes recordings according to the range parameter x_range.

"""
def processSingleRecordingPoint(corine_dir, recording_dir, x_range='all'):    
    dir_files = os.listdir(corine_dir)

    raster_file = os.path.join(corine_dir, list(compress(dir_files, [file.endswith("clip.tif") for file in dir_files]))[0])
    dbf_file = os.path.join(corine_dir, list(compress(dir_files, [file.endswith("vat.dbf") for file in dir_files]))[0])

    # Get DBF File
    dbf_data = DBF(dbf_file)
    dbf_df = pd.DataFrame(iter(dbf_data))

    points = getSoundLocations(recording_dir)
    points["label"] = np.nan

    print("Loading raster.")

    if type(x_range) == str:
        x_range = np.arange(len(points))

    with rasterio.open(raster_file) as tif:
        print("Loaded raster successfully. ")
        raster_crs = tif.crs

        # Reproject points
        if points.crs != raster_crs:
            print(f"Reprojecting points due to differing SRS.")
            points = points.to_crs(raster_crs)

        # Get raster resolution to compute pixel size
        x_res, __ = tif.res

        # Needed for plotting
        all_geo_frames = []
        all_joined_frames = []

        for idx, content in points.iloc[x_range].iterrows():
            try:
                image_data, transformed_data = mask(
                        dataset=tif,
                        shapes=[content.geometry],
                        crop=True,
                        all_touched=True 
                    )
            except:
                print("Could not clip to raster. Assigning zero")
                points.loc[points.index==idx, 'label'] = np.nan
                continue

            print(f"Clipped raster to the point. \nGenerating individual pixel.")

            # Get shape
            rows, cols = image_data.shape[1:]
            # print(rows, cols)
            # Flatten image to a single band
            pixel_values = image_data[0].flatten()

            # Create arrays of all row and column indices
            col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))

            xs, ys = rasterio.transform.xy(transformed_data, row_indices.flatten(), col_indices.flatten())

            # print(xs, ys)

            geometries = gpd.points_from_xy(xs, ys)

            # Create geodataframe
            geo_df = gpd.GeoDataFrame(
                pd.DataFrame({
                    'value': pixel_values,
                }),
                geometry=geometries,
                crs=raster_crs
            )

            if geo_df.value[0]<0:
                points.loc[points.index==idx, 'label'] = np.nan
                continue

            # If squares are not explicitly needed, comment this out
            geo_df['geometry'] = geo_df.buffer(x_res / 2, cap_style="square")

            # print(geo_df)

            all_geo_frames.append(geo_df)

            joined_df = geo_df.merge(dbf_df, left_on = "value", right_on="Value")

            all_joined_frames.append(joined_df)

            # How to handle identical values??, e.g. recording 402
            print(joined_df)
            weighted_class = joined_df["CODE_18"].values[0]


            print(f"Assigned class {weighted_class} to point {idx}")

            points.loc[points.index==idx, 'label'] = weighted_class

    return points, dbf_df, all_geo_frames, all_joined_frames, raster_crs

def computeChangeFrame(frame, frame_single, raster_crs="EPSG:3035"):
    change_frame = frame_single.join(frame, rsuffix="_drop")
    change_frame.drop(columns=["id_drop", "geometry_drop"], inplace=True)
    change_frame["change"] = change_frame["label"].astype(str) != change_frame["label_drop"].astype(str)
    change_frame["change_classes"] = change_frame["label"].astype(str) + "-" +change_frame["label_drop"].astype(str)
    change_frame["interclass_change"] = list(map(lambda x: x[0]!=x[4], change_frame["change_classes"]))
    change_gframe = gpd.GeoDataFrame(change_frame, geometry="geometry", crs=raster_crs)
    return change_gframe

### Data augmentation ###
# Adapted from https://gist.github.com/zcaceres/d2ac50c146fd95df03a8e1c56a7d6f4e
def add_white_noise(signal, noise_scl=0.005, **kwargs):
    noise = torch.randn(signal.shape[0]) * noise_scl
    return signal + noise

def modulate_volume(signal, lower_gain=.1, upper_gain=1.2, **kwargs):
    modulation = random.uniform(lower_gain, upper_gain)
    return signal * modulation

def random_cutout(signal, pct_to_cut=.15, **kwargs):
    """Randomly replaces `pct_to_cut` of signal with silence. Similar to grainy radio."""
    copy = signal.clone()
    sig_len = signal.shape[0]
    sigs_to_cut = int(sig_len * pct_to_cut)
    for i in range(0, sigs_to_cut):
        cut_idx = random.randint(0, sig_len - 1)
        copy[cut_idx] = 0
    return copy

def pitch_warp(signal, sr=16000, sr_divisor=2, **kwargs):
    down_sr = sr // sr_divisor
    resample_down = T.Resample(orig_freq=sr, new_freq=down_sr).to(signal.device, signal.dtype)
    resample_up = T.Resample(orig_freq=down_sr, new_freq=sr).to(signal.device, signal.dtype)
    return resample_up(resample_down(signal))

def apply_augmentation_transforms(signal, tfm):
    # Ensure tfm is a list of transforms
    tfms = tfm if isinstance(tfm, list) else [tfm]
    
    ret = signal
    for t in tfms:
        # If the transform is a Torchaudio/NN module, 
        # sync it to the signal's device and dtype
        if isinstance(t, torch.nn.Module):
            t.to(signal.device, signal.dtype)
        
        ret = t(ret)
    return ret

def create_data_augmentation(file_path, output_dir):
    try:
        target_path = output_dir / (file_path.stem + "_da.pt")

        # Skip the file already exists
        if target_path.exists():
            return True
        
        wave = loadPT(file_path)

        audio_tensor = apply_augmentation_transforms(wave, [modulate_volume, pitch_warp, add_white_noise]).bfloat16()
        # Save tensor
        torch.save(audio_tensor, target_path)
        
        return True
    except Exception as e:
        print(e)
        return False
    

def get_noise_profile(audio_tensor, sr, window_duration=1.0):
    """
    Finds the quietest segment using vectorized Torch operations (No For-Loops).
    """
    window_samples = int(window_duration * sr)
    # Calculating in strides speeds up the processing
    stride = int(sr * 0.1)

    # If audio is too short, return it all
    if audio_tensor.shape[-1] <= window_samples:
        return audio_tensor

    analysis_wave = audio_tensor[0] if audio_tensor.ndim > 1 else audio_tensor

    # Shape: [num_windows, window_samples]
    windows = analysis_wave.unfold(0, window_samples, stride)

    energies = torch.sqrt(torch.mean(windows**2, dim=1))

    min_idx = torch.argmin(energies).item()
    
    start = min_idx * stride
    end = start + window_samples

    if audio_tensor.ndim > 1:
        return audio_tensor[:, start:end]
    return audio_tensor[start:end]
    
def denoise_data(file_path, output_dir, sampling_rate=16000, window_duration=2.5):
    try:
        target_path = output_dir / (file_path.stem + "_dn.pt")

        # Skip the file if it already exists
        if target_path.exists():
            return True
        
        # Load wave
        wave = torch.load(file_path, map_location="cpu") 

        noise_part = get_noise_profile(wave, sampling_rate, window_duration)


        wave_np = wave.numpy()
        noise_np = noise_part.numpy()

        # Reduce noise
        reduced_noise = nr.reduce_noise(
            y=wave_np, 
            sr=sampling_rate, 
            y_noise=noise_np, 
            n_fft=4096,
            hop_length=204, # Approx 95% overlap (4096 * 0.05)
            prop_decrease=1.0
        )

        # Convert back
        reduced_noise_tensor = torch.from_numpy(reduced_noise).bfloat16()
        torch.save(reduced_noise_tensor, target_path)
        
        return True
    except Exception as e:
        # Returning the error string helps debugging
        return str(e)

def process_and_save_as_pt(file_path, output_dir,target_sr=44100, target_loudness=-16):
    """
    Decodes audio via FFmpeg, converts to Tensor, and saves to disk.
    """
    try:
        target_path = output_dir / (file_path.stem + ".pt")
        
        # Skip the file already exists
        if target_path.exists():
            return True

        # Load flac file and process it 
        out, _ = (
            ffmpeg
            .input(str(file_path))
            # Normalize the Loudness
            .filter('loudnorm', i=target_loudness, tp=-1.0, lra=11)
            # ac=1: Convert to mono, ar: sample at target sr, 'f32le': output 32-bit float Little Endian
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=target_sr)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        # Convert bytes to numpy array
        audio_np = np.frombuffer(out, np.float32)

        # Convert to 16-bit Float tensor
        audio_tensor = torch.from_numpy(audio_np.copy()).to(torch.float16)

        # Save tensor
        torch.save(audio_tensor, target_path)
        
        return True
    except Exception as e:
        print(e)
        return False