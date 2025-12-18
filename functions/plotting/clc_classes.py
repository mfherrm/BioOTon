import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pycirclize import Circos
from pycirclize.parser import Matrix

def plotCLCSuperclasses(frame, method="majority vote", fig = None, ax = None):
    # Aggregate by superclass
    counts = frame.loc[pd.to_numeric(frame.label) > 0, "label"].astype(str).str[0].astype(int).value_counts().sort_index()

    # To make this a standalone function
    if (fig== None) | (ax == None):
        fig, ax = plt.subplots(figsize=(8, 6))

    # Build bars manually to get the values later on
    bars = ax.bar(counts.index, counts.values, color='skyblue', edgecolor='black', log=True, width=0.7)

    # Get bar heights to control the minor label format
    bar_heights = [int(5*round(rect.get_height()/5)) for rect in bars]

    # Set xticks
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1\nArtificial", "2\nAgriculture", "3\nForest", "4\nWetlands", "5\nWater"])

    # Set minor yticks and format both minor and major
    ax.set_yticks(bar_heights, minor=True)
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    formatter = ticker.StrMethodFormatter('{x:,.0f}')
    ax.tick_params(axis='y', which='minor', labelsize=7)
    ax.set_ylim(top=ax.get_ylim()[1] * 2) 

    # Set label and title
    ax.set_xlabel("CLC Superclass", fontsize=12)
    ax.set_ylabel("Frequency (Log Scale)", fontsize=12)
    ax.bar_label(bars, padding=3, fontsize=10, fontweight='bold')
    ax.set_title(f"CLC Superclass distribution ({method})", fontsize=14, pad=15)


    plt.grid(axis='y', which='both', linestyle='--', alpha=0.3)

    return fig


"""
    This method ignores subclasses of CLC classes 4 and 5
"""
def plotCLCSubclasses(frame, method="majority vote", fig = None, ax = None):
    # Aggregate CLC by level 2 subclass
    count = frame.loc[pd.to_numeric(frame.label) > 0, "label"].astype(str).str[:2].astype(int).value_counts().sort_index()
    
    # Filter for the first three superclasses
    count = count[count.index < 40]
    
    # To make this a standalone function
    if (fig== None) | (ax == None):
        fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted figsize for a single plot

    # Build bars manually to get the values later on
    bars = ax.bar(count.index, count.values, color='skyblue', edgecolor='black', log=True, width=0.7)

    # Get bar heights to control the minor label format
    bar_heights = [int(5*round(rect.get_height()/5)) for rect in bars]

    # Remove some of the labels that are too close to the major ticks
    for b in bar_heights:
        majors= 10 ** np.arange(6)
        for i in majors:
            if (b/i > 1) & (b/i < 1.1):
                bar_heights.remove(b)
    # Set xticks
    ax.set_xticks(count.index.values)
    ax.set_xticklabels(
        ["1.1\nUrban", "1.2\nIndustrial", "1.3\nConstruction", "1.4\nParks", 
        "2.1\nArable Land", "2.2\nPerm. Crops", "2.3\nPastures", "2.4\nHeterogen Agric.",
        "3.1\nForest", "3.2\nShrub", "3.3\nOpen Spaces"],
        rotation=45, 
        ha='right',
        fontsize=10,
        rotation_mode="anchor"
    )

    # Set minor yticks and format both minor and major
    ax.set_yticks(bar_heights, minor=True)
    fmt = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.yaxis.set_minor_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='y', which='minor', labelsize=8, colors='#555555')
    # Increase top margin to prevent bar labels from cutting off
    ax.set_ylim(top=ax.get_ylim()[1] * 5)


    # Set label and title
    ax.set_xlabel("CLC Subclass", fontsize=12, labelpad=10)
    ax.set_ylabel("Frequency (Log Scale)", fontsize=12)
    ax.bar_label(bars, padding=3, fontsize=9, fontweight='bold', 
                labels=[f'{int(x):,}' for x in count.values])
    ax.set_title(f"CLC Subclass distribution ({method})", fontsize=14, pad=20)

    plt.grid(axis='y', which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()

    return fig

def plotCLCClasses(frame, method="majority vote"):
    # --- Create Figure with 2 Subplots ---
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Plot superclasses
    superclass = plotCLCSuperclasses(frame, fig=fig, ax=ax[0], method=method)

    subclass = plotCLCSubclasses(frame, fig=fig, ax=ax[1], method=method)

    # --- Final Adjustments ---
    plt.tight_layout()
    return fig

def plotCLCChanges(change_gframe, fig = None, ax = None):    
    # To make this a standalone function
    if (fig== None) | (ax == None):
        fig, ax = plt.subplots(figsize=(12, 8))

    # Unchanged centroids
    no_changes = change_gframe[change_gframe["change"] == False].copy()
    if not no_changes.empty:
        no_changes['geometry'] = no_changes.geometry.centroid
        no_changes.plot(ax=ax, color='green', markersize=5, alpha=0.5, label='Unchanged')

    # Changed centroids
    changes = change_gframe[change_gframe["change"] == True].copy()
    if not changes.empty:
        changes['geometry'] = changes.geometry.centroid
        changes.plot(ax=ax, color='red', markersize=10, alpha=0.8, label='Changed')


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

    # Rotate X-Axis ticks to make space for all of them
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    # Calculate positions for Easting and Northing
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_pos = xmax + (xmax - xmin) * 0.01
    y_pos_n = ymax + (ymax - ymin) * 0.02
    y_pos_e = ymin - (ymax - ymin) * 0.0125

    
    # Add Easting
    ax.text(
        x_pos, 
        y_pos_e,
        "E",
        ha='left',
        va='top',
        fontsize=11,
        fontweight='bold',
        clip_on=False
    )

    # Add Northing
    ax.text(
        x_pos, 
        y_pos_n,
        "N",
        ha='left',
        va='top',
        fontsize=11,
        fontweight='bold',
        clip_on=False
    )

    # Add title
    ax.set_title("Geographic Distribution of Changes", fontsize=14)
    ax.legend(loc='lower left')

    return fig

def plotChangeChord(change_gframe, fig= None, ax = None):
    # Select only data that changed
    changes_only = change_gframe[change_gframe["change"] == True].copy()

    # To make this a standalone function
    if (fig== None) | (ax == None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="polar") 


    # Aggregate and count changes
    fromto_table_df = (
        changes_only.groupby(["label_drop", "label"])
        .size()
        .reset_index(name="value")
    )
    fromto_table_df.columns = ["from", "to", "value"]

    # Filter out occurences where less than 25 instances changed
    fromto_table_df = fromto_table_df[fromto_table_df["value"] > 24]

    # Generate matrix in order to feed it to circos
    matrix = Matrix.parse_fromto_table(fromto_table_df)

    # Set circos plot settings
    circos = Circos.initialize_from_matrix(
        matrix,
        space=5,
        cmap="tab20",
        ticks_interval=75,
        ticks_kws=dict(label_orientation="vertical", label_size=5.5),
        label_kws=dict(size=10, r=110),
        link_kws=dict(direction=1, ec="black", lw=0.2, alpha=0.6),
    )

    # Hide axes
    ax.axis('off')
    # Plot
    fig = circos.plotfig(ax=ax)

    # Set title
    fig.suptitle("CLC Label changes (Counts > 25)", fontsize=16, fontweight="bold", y=0.98)

    return fig


def plotChangeAnalysis(change_gframe):
    # Create the Figure and Subplots
    fig = plt.figure(figsize=(22, 10))

    # Standard subplot for the Map
    ax_map = fig.add_subplot(121) 

    # POLAR subplot for the Chord Diagram
    ax_chord = fig.add_subplot(122, projection="polar") 


    plotCLCChanges(change_gframe, fig = fig, ax=ax_map)

    plotChangeChord(change_gframe, fig= fig, ax = ax_chord)

    ax_chord.set_title("CLC Class Transitions (at least 25 changes)", fontsize=14, pad=40)

    # Final Layout Adjustments
    fig.suptitle("Corine Land Cover (CLC) Change Analysis", fontsize=18, fontweight="bold", y=0.98)
    plt.subplots_adjust(wspace=0.3) # Add space between the two plots
    
    return fig