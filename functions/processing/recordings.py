
from dbfread import DBF
from itertools import compress
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
from rasterio.mask import mask
from functions.processing.retrieval import getSoundLocations
"""
    Processes recordings according to the range parameter x_range.

"""
def processSingleRecording(corine_dir, recording_dir, distance = 200, x_range='all'):
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
            weighted_frame = (5 * grouped_dfs[0]).add((3 * grouped_dfs[1]), fill_value=0).add((1 * grouped_dfs[2]), fill_value=0)
            print("Weighted classes:")#, weighted_frame)

            # Ignore NAN
            weighted_frame.loc[weighted_frame.index==-128, 'geometry'] = 0
            all_weighted_frames.append(weighted_frame)
            print(weighted_frame)

            # How to handle identical values??, e.g. recording 402
            weighted_class = weighted_frame.idxmax().item()


            print(f"Assigned class {weighted_class} to point {idx}")

            

            points.loc[points.index==idx, 'label'] = weighted_class

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