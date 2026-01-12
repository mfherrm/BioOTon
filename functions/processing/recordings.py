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
import torchaudio.functional as F
import torchaudio.transforms as T

from functions.processing.retrieval import getSoundLocations
from functions.processing.retrieval import loadPT

"""
    

"""
def processSingleRecording(corine_dir : str, recording_dir : str, distance : float = 200.0, x_range = 'all', mode : str = "simple"):
    """
        Processes recordings according to the range parameter x_range.

        Input:
            corine_dir : str - directory of the tiff file
            recording_dir : str - directory of the recordings
            distance : float - distance of the outer most buffer
            x_range - number of points to use, set 'all' to use all points 
            mode : str - which weighting mode to use, e.g., simple or isdw (inverse squared distance weighting)

        Output: 
            points - dataframe containing all points and their labels
            dbf_df - dataframe containing the CORINE dbf file
            all_geo_frames - all geo frames (one for each buffer)
            all_filtered_frames - all filtered frames (only relevant pixels, one for each buffer)
            all_grouped_frames - all grouped frames (grouped according to their CLC class, one for each buffer)
            all_weighted_frames - all weighted frames (weighted according to their CLC class and the mode, one for each buffer)
            raster_crs - SRS of the raster
    """
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
    """
        Processes recordings according to the range parameter x_range.

        Input:
            corine_dir : str - directory of the tiff file
            recording_dir : str - directory of the recordings
            x_range - number of points to use, set 'all' to use all points 

        Output:
            points - dataframe containing all points and their labels
            dbf_df - dataframe containing the CORINE dbf file
            all_geo_frames - all geo frames (one for each buffer)
            all_joined_frames - all filtered frames (one for each buffer)
            raster_crs - SRS of the raster
    """
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

def computeChangeFrame(frame, frame_single, raster_crs : str ="EPSG:3035"):
    """
        Processes recordings according to the range parameter x_range.

        Input:
            frame - dataframe containing majority voting points
            frame_single - dataframe containing direct assignment points
            raster_crs - SRS of the raster

        Output:
            change_gframe - geodataframe containing changes and change direction for all points (regardless of if they changed or not)
    """
    change_frame = frame_single.join(frame, rsuffix="_drop")
    change_frame.drop(columns=["id_drop", "geometry_drop"], inplace=True)
    change_frame["change"] = change_frame["label"].astype(str) != change_frame["label_drop"].astype(str)
    change_frame["change_classes"] = change_frame["label"].astype(str) + "-" +change_frame["label_drop"].astype(str)
    change_frame["interclass_change"] = list(map(lambda x: x[0]!=x[4], change_frame["change_classes"]))
    change_gframe = gpd.GeoDataFrame(change_frame, geometry="geometry", crs=raster_crs)
    return change_gframe

### Data augmentation ###
# Adapted from https://gist.github.com/zcaceres/d2ac50c146fd95df03a8e1c56a7d6f4e
def add_white_noise(signal, snr : float = 30.0):
    """
        Adds white noise to a signal.

        Input:
            signal - the signal to be processed
            snr : float - in db, use ~ 30 for low noise
        Output:
            tensor - tensor overlayed with white noise
    """
    signal = signal.unsqueeze(0)
    noise = torch.randn_like(signal)

    noisy_signal = F.add_noise(signal, noise, torch.tensor([snr]))
    
    return noisy_signal


def speed_up(signal, orig_frequency : int = 16000, factor : float = 1.15, **kwargs):
    """
        Speeds up / slows down a signal.

        Input:
            signal - the signal to be processed
            orig_frequency : int - frequency of the signal
            factor : float - speed factor

        Output:
            tensor - sped up / slowed down tensor
    """
    speed = T.Speed(orig_frequency, factor)
    speed = speed.to(dtype=signal.dtype, device=signal.device)
    sped_up_signal = speed(signal)[0]
    return sped_up_signal

def cut_off_edge(signal):
    """
        Cuts the left / right edge off a signal.

        Input:
            signal - the signal to be processed

        Output:
            tensor - tensor with cut off edge
    """
    coin_flip = np.random.randint(2)

    clone = signal.clone()
    offset = 16000 * 10

    if coin_flip == 0:
        clone[..., :-offset] = signal[..., offset:].clone()
    else: 
        clone[..., offset:] = signal[..., :-offset].clone()

    return clone

def random_volume_change(signal, sampling_rate : int = 16000, pct : float = 0.1, seconds : float = 1.0):
    """
        Increases / decreases the volume of random parts of a signal.

        Input:
            signal - the signal to be processed
            sampling_rate : int - sampling rate of the signal
            pct : float - percentage of the signal to be affected
            seconds : float - duration of affected change per occurence

        Output:
            tensor - tensor with changed volume
    """
    clone = signal.clone()
    sig_len = signal.shape[-1]
    
    window_size = int(sampling_rate * seconds)
    
    num_iterations = int((sig_len * pct) / window_size)
    
    # Ensure we do at least 1 iteration if pct > 0 and signal is long enough
    num_iterations = max(num_iterations, 1) if pct > 0 else 0

    for i in range(num_iterations):

        if sig_len > window_size:
            cut_idx = np.random.randint(0, sig_len - window_size)

            clone[..., cut_idx : cut_idx + window_size] *= 2
            
    return clone

def random_timeshift(signal):
    """
        Cuts up a signal at random places and rearranges them.

        Input:
            signal - the signal to be processed

        Output:
            tensor - tensor with changed signal places
    """
    max = len(signal)
    borders = np.sort(np.random.rand(2))

    floor = np.floor(max*borders[0]).astype(int)
    ceil = np.floor(max*borders[1]).astype(int)

    shifted = torch.cat([signal[floor:ceil], signal[:floor], signal[ceil:]])

    return shifted

def vertical_blackout(signal, pct : float = 0.1):
    """
        Blacks out the vertical portion of the spectrogram by applying a transformation to a signal.

        Input:
            signal - the signal to be processed
            pct : float - percentage of the signal to be blacked out

        Output:
            tensor - tensor with vertical blackout
    """
    m = len(signal)
    border = np.sort(np.random.rand(1))[0]
    
    floor = np.floor(m*border).astype(int)
    ceil = np.floor(m*(border+pct)).astype(int)

    csignal = signal.clone()
    csignal[floor:ceil] = 0

    return csignal

def horizontal_blackout(signal, sample_rate, center_freq, pct=1250):
    """
        Blacks out the vertical portion of the spectrogram by applying a transformation to a signal.

        Input:
            signal - the signal to be processed
            sample_rate - sample rate of the signal
            center_freq - frequency for the bandpass filter
            pct : float - part of the signal to be blacked out

        Output:
            tensor - tensor with horizontal blackout
    """

    center_freq = np.random.uniform(0.15, 1.0) * center_freq
    
    nyquist = sample_rate / 2
    bandwidth = nyquist * pct
    
    Q = center_freq / max(bandwidth, 1e-6)

    print(Q)
    
    band = F.bandpass_biquad(signal, sample_rate, center_freq, Q=Q)
        
    return signal - band

def modulate_volume(signal, lower_gain : float = .1, upper_gain : float = 1.2, **kwargs):
    """
        Modulates the volume of a signal randomly. This does not work with normalization in spectrograms.

        Input:
            signal - the signal to be processed
            lower_gain : float - strongest decrease of the signal, i.e. to 10% original volume
            upper_gain : float - strongest increase of the signal, i.e. to 120% original volume

        Output:
            tensor - tensor with modulated volume
    """
    modulation = random.uniform(lower_gain, upper_gain)
    return signal * modulation

def random_cutout(signal, pct :float = .15, **kwargs):
    """
        Adds random grain to the signal.

        Input:
            signal - the signal to be processed
            pct : percentage of the signal to be made grainy

        Output:
            tensor - tensor with grain effect
    """
    copy = signal.clone()
    sig_len = signal.shape[0]
    sigs_to_cut = int(sig_len * pct)
    for i in range(0, sigs_to_cut):
        cut_idx = random.randint(0, sig_len - 1)
        copy[cut_idx] = 0
    return copy

def oversample(signal, orig_freq=16000, new_freq=44100):
    """
        Oversamples a signal.

        Input:
            signal - the signal to be processed

        Output:
            tensor - oversampled tensor
    """
    oversampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq).to(dtype=signal.dtype, device=signal.device)
    
    return oversampler(signal)

def pitch_warp(signal, sr : int = 16000, sr_divisor : float = 2.0, **kwargs):
    """
       Performs lossy sampling on a signal.

        Input:
            signal - the signal to be processed
            sr : int - sampling rate of the original signal
            sr_divisor - proportion of the sampling

        Output:
            tensor - lossily sampled tensor
    """
    down_sr = sr // sr_divisor
    resample_down = T.Resample(orig_freq=sr, new_freq=down_sr).to(signal.device, signal.dtype)
    resample_up = T.Resample(orig_freq=down_sr, new_freq=sr).to(signal.device, signal.dtype)
    return resample_up(resample_down(signal))

def apply_augmentation_transforms(signal, tfm):
    """
       Apply all augmentation transforms given to a signal.

        Input:
            signal - the signal to be processed
            tfm - list of all functions to apply
        Output:
            tensor - transformed tensor
    """
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
    """
       Apply given transforms to multiple signals and saves them to the output directory.

        Input:
            file_path - the directory containing the signals to be processed
            output_dir - directory to save the transformed tensors to
    """
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
    

def get_noise_profile(audio_tensor, sr : int, window_duration : float = 1.0):
    """
       Finds the quietest segment to find signal characteristics.

        Input:
            audio_tensor - the signal to be processed
            sr : int - original sampling rate
            window_duration : float - duration of the window to find
        Output:
            tensor - characteristic tensor
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
    
def denoise_data(file_path : str, output_dir : str, sampling_rate : int =16000, window_duration : float =2.5):
    """
       Denoises data by finding characteristic profile, then saves it to the disk.

        Input:
            file_path : str - the directory containing the signals to be processed
            output_dir : str - directory to save the transformed tensors to
            audio_tensor - the signal to be processed
            sampling_rate : int - original sampling rate
            window_duration : float - duration of the window to find
        Output:
            tensor - characteristic tensor
    """
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

def process_and_save_as_pt(file_path : str, output_dir : str, target_sr : int = 44100, target_loudness : float = -16.0):
    """
       Decodes audio via FFmpeg, converts it to a tensor, and saves to the disk.

        Input:
            file_path : str - the directory containing the signals to be processed
            output_dir : str - directory to save the transformed tensors to
            target_sr : int - sampling rate to resample the signal to
            target_loudness : float - target loudness to transform the signal to
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