import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_hex
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import geopandas as gpd


def plotRecord(point_idx : int, dbf_frame, recording_frame, joined_frame, environment_frame, fig = None, ax = None):
    """
        Plots a record and its surrounding CLC classes.

        Input:
            point_idx : int - the point to plot
            dbf_frame - dbd dataframe of the original tiff, to specify the original colors 
            recording_frame - the dataframe containing the point data 
            joined_frame - dataframe containing the relevant pixels 
            environment_frame - geodataframe with all pixels 
            fig - provided to plot into an existing figure 
            ax - provided to plot into an existing figure

        Output: 
            fig - figure containing the plot
    """
    recording_frame= recording_frame.dropna()
     # To make this a standalone function
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Set Super Title as recording number
    fig.suptitle(f"CLC class assignment for point {recording_frame.iloc[point_idx].id}", fontsize=13, fontweight='bold')


    # This value frame is needed for the CLC polygons and their legend
    # Duplicates in joined_frame have to be dropped in order to facilitate a successful merge
    unique_jdf = pd.concat(joined_frame[point_idx]).drop_duplicates(subset=['value'], keep='first').drop(columns=['geometry', 'Value'])
    val_frame = environment_frame[point_idx].merge(unique_jdf, on = "value", how="left", suffixes=('', '_r'))
    # val_frame = environment_frame.join(joined_frame, on = "value", how="left", rsuffix='_r')

    # Generate a dict using the same colors as the original CLC tiff
    color_dict = {
        code: to_hex((R, G, B)) # Matplotlib expects normalized (0.0 to 1.0) RGB first
        for code, R, G, B in zip(dbf_frame.CODE_18.values, dbf_frame.Red.values, dbf_frame.Green.values, dbf_frame.Blue.values)
    }

    # Needed to map the values to the legend descriptors
    codes = val_frame[['CODE_18', 'LABEL3']].drop_duplicates().sort_values('CODE_18')['CODE_18'].tolist()
    labels = val_frame[['CODE_18', 'LABEL3']].drop_duplicates().sort_values('CODE_18')['LABEL3'].replace(float("nan"), "No data").tolist()

    # Maps legend labels and colors
    legend_handles = [
        mpatches.Patch(
            color=color_dict.get(code, 'gray'), 
            label=label
        ) 
        for code, label in zip(codes, labels)
    ]

    # Plot all CLC classes in the area
    val_frame.plot(ax=ax, legend= False, column="CODE_18", categorical=True, color = [color_dict.get(c, 'gray') for c in val_frame['CODE_18']])

    # Generate Legend
    ax.legend(
        handles=legend_handles, 
        title="CLC Class", 
        loc='upper right', 
        frameon=True,
        fontsize=9
    )

    # Plot annotations, mainly index to the CLC classes
    for idx, row in val_frame.iterrows():
        # Get square center
        centroid_x = row.geometry.centroid.x
        centroid_y = row.geometry.centroid.y
        ax.annotate(
            text=f"{str(idx)}",
            # Comment out below unless you want to do a sanity check
            #text=f"{str(idx)} (c: {(row.CODE_18)})",
            xy=(centroid_x, centroid_y),
            ha='center',
            va='center',
            fontsize=9,
            color='white',
            fontweight='bold'
        )


    # Disable scientific format and format ticks so that they use , to separate steps of 1000
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    formatter = ticker.StrMethodFormatter('{x:,.0f}')

    # Apply formatting
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add y axis label
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()


    # Add y axis label
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_pos = xmax + (xmax - xmin) * 0.01
    y_pos = ymax + (ymax - ymin) * 0.02

    ax.text(
        x_pos, 
        y_pos,
        "N",
        ha='left',
        va='top',
        fontsize=11,
        fontweight='bold',
        clip_on=False
    )

    # Add x axis label
    y_pos = ymin - (ymax - ymin) * 0.0125

    ax.text(
        x_pos, 
        y_pos,
        "E",
        ha='left',
        va='top',
        fontsize=11,
        fontweight='bold',
        clip_on=False
    )

    ax.scatter(recording_frame.iloc[point_idx].geometry.x, recording_frame.iloc[point_idx].geometry.y,  color='black', edgecolor='white', zorder=10)

    # Generate a grid at every 100m
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid()

    plt.tight_layout()

    return fig


# Only problem with this index scheme is that the indexes are always dependent on how many points were processed previously
def plotRecordWithBuffers(point_idx, dbf_frame, recording_frame, joined_frame, environment_frame, num_buffers : int = 3, distance : float = 200.0, crs : str = "EPSG:3035", fig = None, ax = None):
    """
        Plots a record and its surrounding CLC classes as well as its buffers.

        Input:
            point_idx : int - the point to plot
            dbf_frame - dbd dataframe of the original tiff, to specify the original colors
            recording_frame - the dataframe containing the point data
            joined_frame - dataframe containing the relevant pixels
            environment_frame - geodataframe with all pixels
            num_buffers : int - number of buffers to create
            distance : float - distance of the outer most buffer
            crs : str - SRS to create the buffer in 
            fig - provided to plot into an existing figure 
            ax - provided to plot into an existing figure

        Output: 
            fig - figure containing the plot
    """
    recording_frame= recording_frame.dropna()
    # To make this a standalone function
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    plotRecord(point_idx=point_idx, dbf_frame = dbf_frame, recording_frame=recording_frame, joined_frame=joined_frame, environment_frame=environment_frame, fig=fig, ax=ax)

    # Generate buffer colors
    if num_buffers < 4:
        colors = ["red", "orange", "green"]
    else: 
        colors = plt.colormaps["RdYlGn"]

    # Plot buffers
    buffers = [gpd.GeoSeries(recording_frame.iloc[point_idx].geometry, crs=crs).buffer(d) for d in np.arange(start=distance/num_buffers, step=distance/num_buffers, stop= distance+1)]
    for idx, x in enumerate(list(reversed(buffers))):
        if num_buffers < 4:
            x.plot(
                ax=ax, 
                facecolor='none',
                hatch='X', 
                edgecolor=colors[idx], 
                alpha=1.0,
                linewidth=2,
                zorder=5
            )
        else:
            x.plot(
                ax=ax, 
                facecolor='none',
                hatch='X', 
                edgecolor=colors(idx*0.1), 
                alpha=1.0,
                linewidth=2,
                zorder=5
            )

"""
    Generates plot for given recording index.
    
"""
def plotSingleRecordingAnalysis(point_idx, dbf_frame, recording_frame, joined_frame, environment_frame, grouped_frames, weighted_frames, num_buffers=3, distance =200, raster_crs="EPSG:3035"):
    """
        Plots a record and its surrounding CLC classes as well as its buffers on the left and a summary of classes per buffer on the right.
        This only works if the length of recording_frame is identical to the length of environment-frame / joined_frame etc.
        If this is not the case use plotSingleRecording from /plotting/sliced_records

        Input:
            point_idx : int - the point to plot
            dbf_frame - dbd dataframe of the original tiff, to specify the original colors
            recording_frame - the dataframe containing the point data
            joined_frame - dataframe containing the relevant pixels
            environment_frame - geodataframe with all pixels
            grouped_frames - grouped dataframes with groups being decided by CLC classses
            weighted_frames -  grouped dataframes weighted by the chosen method
            num_buffers : int - number of buffers to create
            distance : float - distance of the outer most buffer
            crs : str - SRS to create the buffer in 
            fig - provided to plot into an existing figure 
            ax - provided to plot into an existing figure

        Output: 
            fig - figure containing the plot
    """
    recording_frame= recording_frame.dropna()
    # Generate plot
    fig, ax= plt.subplots(1, 2, figsize=(20, 8))

    plotRecordWithBuffers(point_idx=point_idx, dbf_frame=dbf_frame, recording_frame=recording_frame, joined_frame=joined_frame, environment_frame=environment_frame, num_buffers=num_buffers, distance=distance, crs= raster_crs, fig=fig, ax=ax[0])

    # This dataframe is needed to generate the class pixel count table 
    combined_df = pd.concat(map(pd.DataFrame.reset_index, grouped_frames[point_idx]), ignore_index=False, keys=[0, 1, 2], names=['df', "idx"])
    cdf = combined_df.reset_index().rename(columns={"df":"Buffer Ring", "CODE_18":"CORINE Class Code", "geometry":"Count"}).drop(columns="idx")

    # Used to label the pixel count table
    combined_headers = cdf.columns.tolist()

    # Add title as text
    ax[1].text(0.5, 0.98, "Pixel Count by Class and Buffer", 
            transform=ax[1].transAxes, ha='center', fontsize=12, fontweight='bold')

    # Draw the table
    
    table1 = ax[1].table(
        cellText=cdf.values,
        colLabels=combined_headers,
        colLoc='center',
        cellLoc='center',
        loc='top',
        bbox=[0.1, 0.65, 0.8, 0.3]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)

    # Highlight header
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dcdcdc")
            cell.set_text_props(fontweight='bold')

    # This frame is needed for the pixel weight table    
    w_frame = weighted_frames[point_idx].reset_index().rename(columns={"CODE_18":"CORINE Class Code", "geometry":"Weighted score"})
    weighted_headers = w_frame.columns.tolist()

    # Add title as text
    ax[1].text(0.5, 0.575, "Weighted classes", 
            transform=ax[1].transAxes, ha='center', fontsize=12, fontweight='bold')
    
    # Draw the table
    table2 = ax[1].table(
        cellText=w_frame.values,
        colLabels=weighted_headers,
        colLoc='center',
        cellLoc='center',
        loc='top', 
        bbox=[0.1, 0.425, 0.8, 0.125] # Adjust position
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    # Highlight header
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dcdcdc")
            cell.set_text_props(fontweight='bold')
            cell.set_height(0.029)

    # Hide axes
    ax[1].axis('off')

    # Give weighting explanation
    weight_text = "Classes in ring 0 are weighted by a factor of 5, by 3 in ring 1 and by 1 in ring 2."

    # Declare selected label
    ax[1].text(0.11, 0.4, weight_text, 
            transform=ax[1].transAxes, ha='left', va='center', fontsize=9, color="grey")

    ax[1].text(0.1, 0.2, f"Selected class: {recording_frame.iloc[point_idx].label}", fontsize= 12, fontweight = "bold")


    plt.tight_layout()

    return fig