
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
import pandas as pd
import subprocess
from CIRESA.utils import suppress_output


def plot_spacecraft_carrington(spacecraft, rlim = 1.2, axes = None):
    #matplotlib.use('Agg')
    if 'CARR_LON_RAD' not in spacecraft:
        spacecraft['CARR_LON_RAD'] = spacecraft['CARR_LON']/180*3.14159
    
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=True, subplot_kw={'projection': 'polar'})

    s=10

    #sns.scatterplot(data=sim[sim['ITERATION']<i], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    #sns.scatterplot(data=sim_re[sim_re['ITERATION'] > (sim_re.iloc[0]['ITERATION'] - i)], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)

    sns.scatterplot(data=spacecraft, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), linewidth=0, legend=False)


    #axes.set_xlim([1.0,3.8])
    axes.set_rlim([0, rlim])
    axes.set_xlabel('')
    axes.set_ylabel('                   longitude [°]')
    axes.text(0.6, 0.5, 'r [AU]')
    #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
    axes.set_axisbelow(False)
    axes.grid(True, which='both', zorder=3, linewidth=0.2)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='flare', norm=plt.Normalize(vmin=400, vmax=600))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05, shrink=0.4, aspect=15)
    cbar.set_label('v [km/s]')


    if axes is None:
            plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
            plt.show()

def plot_CIR_carrington(spacecraft, rlim = 1.2, axes=None):
    #matplotlib.use('Agg')
    spacecraft['CARR_LON_RAD'] = spacecraft['CARR_LON']/180*3.14159

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharex=True, subplot_kw={'projection': 'polar'})

    s=10

    #sns.scatterplot(data=sim[sim['ITERATION']<i], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    #sns.scatterplot(data=sim_re[sim_re['ITERATION'] > (sim_re.iloc[0]['ITERATION'] - i)], x='L', y = 'R', ax = axes, s=3*s, hue='V', palette='flare', hue_norm=(400,600), legend=False, linewidth=0)
    
    sns.scatterplot(data=spacecraft, x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='grey', alpha = 0.1, linewidth=0, legend=False)

    if len(spacecraft[spacecraft['Region']==1])>0:
        sns.scatterplot(data=spacecraft[spacecraft['Region']==1], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='orange', alpha = 0.1, linewidth=0, legend=False)

    if len(spacecraft[spacecraft['Region']==2])>0:
       sns.scatterplot(data=spacecraft[spacecraft['Region']==2], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='red', linewidth=0, legend=False)
        
    if len(spacecraft[spacecraft['Region']==3])>0: 
        sns.scatterplot(data=spacecraft[spacecraft['Region']==3], x='CARR_LON_RAD', y = 'R', ax = axes, s=3*s, color='black', linewidth=0, legend=False)
              


    #axes.set_xlim([1.0,3.8])
    axes.set_rlim([0, rlim])
    axes.set_xlabel('')
    axes.set_ylabel('                   longitude [°]')
    axes.text(0.6, 0.5, 'r [AU]')
    #axes.legend(loc='lower center', bbox_to_anchor=(0.65, 0), ncol=1)
    axes.set_axisbelow(False)
    axes.grid(True, which='both', zorder=3, linewidth=0.2)

    if axes is None:
            plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
            plt.show()


def plot_n_days(df, directory='NDAYPlots', n=10, movie=False, plot_cadence=24, CIR=False):
    matplotlib.use('Agg')  # Use a non-GUI backend for plotting
    if not os.path.exists(directory):
        os.makedirs(directory)

    timerange = [df.index[0], df.index[-1]]
    total_hours = (timerange[1] - timerange[0]).total_seconds() / 3600  # Convert total time to hours

    num_steps = int(total_hours // plot_cadence)  # Number of steps based on the specified hour interval

    for i in range(num_steps - (n * 24) // plot_cadence + 1):  # Ensure not to exceed the range
        print(f'Plot {i} out of {num_steps - (n * 24) // plot_cadence + 1}')

        lower_index = df.index[0] + pd.Timedelta(hours=i * plot_cadence)
        upper_index = lower_index + pd.Timedelta(days=n)  # Plot for 'n' days

        # Slice the DataFrame between lower and upper indices while keeping duplicates
        df_slice = df[(df.index >= lower_index) & (df.index <= upper_index)]

        # Plotting function (assuming it takes a DataFrame slice)
        if CIR:
            plot_CIR_carrington(df_slice)
        else:
            plot_spacecraft_carrington(df_slice)

        # Save the plot
        filename = os.path.join(directory, f'plot_{i:04d}.png')
        plt.savefig(filename, format='png')
        plt.close()  # Close the plot to avoid overlap

    if movie:
        make_movie(directory)

def make_movie(directory):

    print('Preparing movie...')

    # Output video filename
    output_video = os.path.join(directory, 'movie.mp4')

    # FFmpeg command to create a movie from PNG images
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", "10",                # Frames per second
        "-i", os.path.join(directory, 'plot_%4d.png'), # Input image filenames (plot_0.png, plot_1.png, etc.)
        "-c:v", "libx264",                # Video codec (H.264)
        "-pix_fmt", "yuv420p",            # Pixel format
        "-vf", "scale=1280:720",          # Output video resolution
        "-r", "10",                       # Output framerate
        "-y",                             # Overwrite output file if it already exists
        output_video
    ]


    # Run FFmpeg command
    subprocess.run(ffmpeg_cmd, check=True)
    
    print(f'Movie saved to: {output_video}')


from CIRESA import propagation
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from CIRESA import get_coordinates
from sunpy.coordinates import get_horizons_coord


def progressive_model_movie(df_list, directory='Modelmovie'
                            , persistance=10, plot_cadence=24, sim_cadence = '0.5H'
                            , model='ballistic'
                            , CIR=False
                            , HEE = False
                            , back_prop = False
                            , movie=False
                            , rlim = 1.2
                            , COR = 0):
    
    """
    Generates progressive plots for spacecraft data over time, 
    creates model data and optionally creates a movie.
    
    Args:
        df_list (list of pd.DataFrame): List of DataFrames containing spacecraft data.
        directory (str): Directory to save the plots.
        persistance (int): Number of days per plot window.
        movie (bool): Whether to create a movie from the plots.
        plot_cadence (int): Time interval in hours for each step.
        CIR (bool): Whether to plot CIR (Co-rotating Interaction Regions).
        model (str): The propagation model to use ('ballistic' or 'inelastic').
    """

    matplotlib.use('Agg')  # Non-GUI backend for headless plot generation
    
    #SORT
    last_values = [df['CARR_LON_RAD'].iloc[-1] for df in df_list]
    value_df_pairs = list(zip(last_values, df_list))
    sorted_value_df_pairs = sorted(value_df_pairs, key=lambda x: x[0])
    df_list = [df for _, df in sorted_value_df_pairs]

    # Concatenate all input DataFrames
    concat_df = pd.concat(df_list)

    # Create directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Calculate total time range in hours
    timerange = [concat_df.index[0], concat_df.index[-1]]
    total_hours = (timerange[1] - timerange[0]).total_seconds() / 3600

    # Number of steps based on the specified hour interval
    num_steps = int(total_hours // plot_cadence)

    # Ensure that we're not exceeding the available range
    for i in range(num_steps - (persistance * 24) // plot_cadence + 1):
        print(f'Plot {i+1} out of {num_steps - (persistance * 24) // plot_cadence + 1}')

        

        # Define the time window for the current plot (lower and upper index)
        lower_index = concat_df.index[0] + pd.Timedelta(hours=i * plot_cadence)
        upper_index = lower_index + pd.Timedelta(days=persistance)
        
        #SORT AGAIN

        last_values = []

        for df in df_list:
            if upper_index in df['CARR_LON_RAD'].index:
                last_values.append(df['CARR_LON_RAD'].loc[upper_index].iloc[0])
            else:
                last_values.append(np.nan)
        value_df_pairs = list(zip(last_values, df_list))
        sorted_value_df_pairs = sorted(value_df_pairs, key=lambda x: x[0])
        df_list = [df for _, df in sorted_value_df_pairs]

        # Concatenate all input DataFrames
        concat_df = pd.concat(df_list)


        # Slice the in-situ DataFrame for the current time window
        insitu_df_slice = concat_df[(concat_df.index >= lower_index) & (concat_df.index <= upper_index)]

        # Initialize a list to store the simulation DataFrames
        sim_df = []
        
        # Iterate over the provided DataFrames and simulate using the selected model
        for df in df_list:
            df_slice = df[(df.index >= lower_index) & (df.index <= upper_index)]
            
            if not df_slice.empty:
                # Apply the chosen propagation model
                if model == 'inelastic':
                    sim = suppress_output(propagation.inelastic_radial, df_slice, sim_cadence, COR=COR)
                elif model == 'ballistic':
                    sim = suppress_output(propagation.ballistic, df_slice, sim_cadence)
                else:
                    print('Unsupported model:', model)
                    return False
                if back_prop:
                    sim_back = suppress_output(propagation.ballistic_reverse, df_slice, sim_cadence)
                    sim = pd.concat([sim_back, sim])
                
                sim_df.append(sim)


        # Concatenate the in-situ data and simulation results for plotting
        plot_df = pd.concat([insitu_df_slice] + sim_df)
        analyze_df = pd.concat([insitu_df_slice] + sim_df)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 14), subplot_kw={'projection': 'polar'})
        ax.remove()
        axes = fig.add_axes([0.1, 0.22, 0.8, 0.8], projection='polar')
        
        carr = suppress_output(get_coordinates.get_carrington_longitude, upper_index)
        Earth_inert = suppress_output(get_horizons_coord, '3', upper_index)
        stereo_a_inert = suppress_output(get_horizons_coord, 'STEREO-A', upper_index)
        solo_inert = suppress_output(get_horizons_coord, 'Solar Orbiter', upper_index)
        psp_inert = suppress_output(get_horizons_coord, 'PSP', upper_index)

        if HEE:
            plot_df['CARR_LON_RAD'] = plot_df['CARR_LON_RAD'] - carr/180*np.pi# + Earth_inert.heliocentricHEE.lon.value/180*np.pi
            axes.spines['polar'].set_visible(True)

        # Call the appropriate plot function (CIR or spacecraft Carrington plot)
        if CIR:
            plot_CIR_carrington(plot_df, rlim=rlim, axes=axes)
        else:
            plot_spacecraft_carrington(plot_df, rlim=rlim, axes=axes)
        
        if HEE:
            sns.scatterplot(x= [0], y = [1], ax = axes, s=100, color='blue', linewidth=0, legend=False)
            sns.scatterplot(x= [(psp_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [psp_inert.radius.value], ax = axes, s=100, color='red', linewidth=0, legend=False)
            sns.scatterplot(x= [(solo_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [solo_inert.radius.value], ax = axes, s=100, color='yellow', linewidth=0, legend=False)
            sns.scatterplot(x= [(stereo_a_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [stereo_a_inert.radius.value], ax = axes, s=100, color='black', linewidth=0, legend=False)
        
        else:
            sns.scatterplot(x= carr/180*np.pi, y = [1], ax = axes, s=50, color='blue', linewidth=0, legend=False)
            sns.scatterplot(x= (psp_inert.lon.value - Earth_inert.lon.value - carr)/180*np.pi, y = [psp_inert.radius.value], ax = axes, s=50, color='red', linewidth=0, legend=False)
            sns.scatterplot(x= (solo_inert.lon.value - Earth_inert.lon.value - carr)/180*np.pi, y = [solo_inert.radius.value], ax = axes, s=50, color='yellow', linewidth=0, legend=False)
            sns.scatterplot(x= (stereo_a_inert.lon.value - Earth_inert.lon.value - carr)/180*np.pi, y = [stereo_a_inert.radius.value], ax = axes, s=50, color='black', linewidth=0, legend=False)
                   

        au = propagation.cut_from_sim(analyze_df)
        # Add a second set of normal (Cartesian) axes below the polar plot
        ax_timeseries = fig.add_axes([0.1, 0.05, 0.8, 0.2])  # [left, bottom, width, height]
        au['CARR_LON'] = (au['CARR_LON_RAD'] * 180/np.pi)%360
        if HEE:
            au['CARR_LON_RAD'] = au['CARR_LON_RAD'] - carr/180*np.pi# + Earth_inert.heliocentricHEE.lon.value/180*np.pi
            au['CARR_LON'] = au['CARR_LON_RAD'] * 180/np.pi
            au['CARR_LON'] = np.where(au['CARR_LON'] > 180, au['CARR_LON'] - 360, au['CARR_LON'])
            
        if CIR:
            au['CARR_LON'] = au['CARR_LON_RAD']
            if len(au[au['Region']==1])>0:
                sns.scatterplot(data=au[au['Region']==1], x='CARR_LON', y = 'V', ax = ax_timeseries, s=5, color='orange', linewidth=0, legend=False)

            if len(au[au['Region']==2])>0:
                sns.scatterplot(data=au[au['Region']==2], x='CARR_LON', y = 'V', ax = ax_timeseries, s=5, color='red', linewidth=0, legend=False)
                
            if len(au[au['Region']==3])>0: 
                sns.scatterplot(data=au[au['Region']==3], x='CARR_LON', y = 'V', ax = ax_timeseries, s=5, color='black', linewidth=0, legend=False)
    
        else:       
            sns.scatterplot(data=au, x='CARR_LON', y = 'V', ax = ax_timeseries, s=5, color='black', linewidth=0, legend=False)
                
        if not HEE:
            ax_timeseries.set_xlim(360,0)
            ax_timeseries.vlines(x = carr, ymin=300, ymax=800, color='blue', label='Earth')
            ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value - carr, ymin=300, ymax=800, color='red', label='PSP')
            ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value - carr, ymin=300, ymax=800, color='yellow', label='SolO')
            ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value - carr, ymin=300, ymax=800, color='black', label='STEREO-A')
            ax_timeseries.set_xlabel('CARR_LON')
            
        else: 
            ax_timeseries.set_xlabel('HEE_LON')
            ax_timeseries.set_xlim(180,-180)
            ax_timeseries.vlines(x = 0, ymin=300, ymax=800, color='blue', label='Earth')
            ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value, ymin=300, ymax=800, color='red', label='PSP')
            ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value, ymin=300, ymax=800, color='yellow', label='SolO')
            ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value, ymin=300, ymax=800, color='black', label='STEREO_A')
            
        ax_timeseries.set_ylim(300,800)
        ax_timeseries.set_title('1 AU')
        ax_timeseries.set_ylabel('km/s')
        ax_timeseries.legend(loc='upper right')

        fig.text(0.05, 0.95, upper_index.strftime('%Y-%m-%d %H:%M:%S'))

        # Save the generated plot to a file
        filename = os.path.join(directory, f'plot_{i:04d}.png')
        plt.savefig(filename, format='png')
        plt.close()  # Close the plot to free memory

    # If movie flag is set, generate a movie from the saved plots
    if movie:
        make_movie(directory)
