
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
import pandas as pd
import subprocess
from CIRESA.utils import suppress_output


def plot_spacecraft_carrington(spacecraft, rlim = 1.2, xlim=None, axes = None, s = 10, variable_to_plot = 'V'):
    #matplotlib.use('Agg')
    if 'CARR_LON_RAD' not in spacecraft:
        spacecraft['CARR_LON_RAD'] = spacecraft['CARR_LON']/180*3.14159
    
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=True, subplot_kw={'projection': 'polar'})

    spacecraft[variable_to_plot].dropna(inplace=True)
    vmin=np.min(spacecraft[variable_to_plot])
    vmax=np.max(spacecraft[variable_to_plot])
    #print(vmin, vmax)

    #sns.scatterplot(data=sim[sim['ITERATION']<i], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    #sns.scatterplot(data=sim_re[sim_re['ITERATION'] > (sim_re.iloc[0]['ITERATION'] - i)], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    if variable_to_plot == 'V':
        sns.scatterplot(data=spacecraft, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare', hue_norm=(400,600), linewidth=0, legend=False)
    else: 
        sns.scatterplot(data=spacecraft, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue=variable_to_plot, palette='flare'
                        , hue_norm=(vmin, vmax), linewidth=0, legend=False)
    
    if xlim is not None:
        xlim = [xlim[0] * np.pi / 180, xlim[1] * np.pi / 180]
        axes.set_xlim(xlim)
    axes.set_rlim([0, rlim])
    axes.set_xlabel('')
    axes.set_ylabel('                                             longitude [°]')
    axes.text(0.6, 0.5, 'r [AU]')
    #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
    axes.set_axisbelow(False)
    axes.grid(True, which='both', zorder=3, linewidth=0.2)

    # Add a colorbar
    if variable_to_plot == 'V':
        sm = plt.cm.ScalarMappable(cmap='flare', norm=plt.Normalize(vmin=400, vmax=600))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.4, aspect=15)
        cbar.set_label('v [km/s]')
    else:
        sm = plt.cm.ScalarMappable(cmap='flare', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.4, aspect=15)
        cbar.set_label(variable_to_plot)


    if axes is None:
            plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
            plt.show()
            plt.close()

def plot_CIR_carrington(spacecraft, rlim = 1.2, xlim = None,  axes=None, s=10):
    #matplotlib.use('Agg')
    if 'CARR_LON_RAD' not in spacecraft:
        spacecraft['CARR_LON_RAD'] = spacecraft['CARR_LON']/180*3.14159
    
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=True, subplot_kw={'projection': 'polar'})

    #sns.scatterplot(data=sim[sim['ITERATION']<i], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    #sns.scatterplot(data=sim_re[sim_re['ITERATION'] > (sim_re.iloc[0]['ITERATION'] - i)], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    
    sns.scatterplot(data=spacecraft, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='grey', alpha = 0.1, linewidth=0, legend=False)
    if len(spacecraft[spacecraft['Region']==3])>0: 
        sns.scatterplot(data=spacecraft[spacecraft['Region']==3], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='black', linewidth=0, legend=False)
    
    if len(spacecraft[spacecraft['Region']==2])>0:
       sns.scatterplot(data=spacecraft[spacecraft['Region']==2], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='red', linewidth=0, legend=False)
            
    if len(spacecraft[spacecraft['Region']==1])>0:
        sns.scatterplot(data=spacecraft[spacecraft['Region']==1], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='orange', alpha = 0.1, linewidth=0, legend=False)

  

    if xlim is not None:
        xlim = [xlim[0] * np.pi / 180, xlim[1] * np.pi / 180]
        axes.set_xlim(xlim)
    axes.set_rlim([0, rlim])
    axes.set_xlabel('')
    axes.set_ylabel('                                             longitude [°]')
    axes.text(0.6, 0.5, 'r [AU]')
    #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
    axes.set_axisbelow(False)
    axes.grid(True, which='both', zorder=3, linewidth=0.2)

    if axes is None:
            plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
            plt.show()


def plot_n_days(df, directory='NDAYPlots', persistance=10, rlim = 1.2
                , movie=False, plot_cadence=24, CIR=False):
    matplotlib.use('Agg')  # Use a non-GUI backend for plotting
    if not os.path.exists(directory):
        os.makedirs(directory)

    timerange = [df.index[0], df.index[-1]]
    total_hours = (timerange[1] - timerange[0]).total_seconds() / 3600  # Convert total time to hours

    num_steps = int(total_hours // plot_cadence)  # Number of steps based on the specified hour interval

    for i in range(num_steps - (persistance * 24) // plot_cadence + 1):  # Ensure not to exceed the range
        print(f'Plot {i} out of {num_steps - (persistance * 24) // plot_cadence + 1}')

        lower_index = df.index[0] + pd.Timedelta(hours=i * plot_cadence)
        upper_index = lower_index + pd.Timedelta(days=persistance)  # Plot for 'n' days

        # Slice the DataFrame between lower and upper indices while keeping duplicates
        df_slice = df[(df.index >= lower_index) & (df.index <= upper_index)]

        # Plotting function (assuming it takes a DataFrame slice)
        if CIR:
            plot_CIR_carrington(df_slice, rlim=rlim)
        else:
            plot_spacecraft_carrington(df_slice, rlim=rlim)

        # Save the plot
        filename = os.path.join(directory, f'plot_{i:04d}.png')
        plt.savefig(filename, format='png')
        plt.close()  # Close the plot to avoid overlap

    if movie:
        make_movie(directory)

import os
import subprocess

def rename_files_to_sequence(directory):
    # Get a sorted list of all .png files in the directory
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') and f[0].isalpha()])

    # If no valid image files are found, raise an error
    if not image_files:
        raise ValueError("No alphabetic .png files found in the directory.")

    # Rename files to the format 'plot_0001.png', 'plot_0002.png', etc.
    for idx, filename in enumerate(image_files):
        # Generate the new filename with zero-padded index
        new_filename = f'plot_{idx:04d}.png'
        
        # Construct full path for old and new filenames
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} -> {new_filename}')
    
    print(f"Renaming complete. Files are now named in sequence.")

def make_movie(directory, framerate=10, rename=False):

    print('Preparing movie...')
    framerate = str(framerate)
    # Output video filename
    output_video = os.path.join(directory, 'movie.mp4')

    if rename:
        rename_files_to_sequence(directory)

    # FFmpeg command to create a movie from PNG images
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", framerate,                # Frames per second
        "-i", os.path.join(directory, 'plot_%4d.png'), # Input image filenames
        "-c:v", "libx264",                # Video codec (H.264)
        "-pix_fmt", "yuv420p",            # Pixel format
        "-r", framerate,                       # Output framerate
        "-y",                             # Overwrite output file if it already exists
        output_video
    ]
    

    # Run FFmpeg command
    subprocess.run(ffmpeg_cmd, check=True)
    
    print(f'Movie saved to: {output_video}')