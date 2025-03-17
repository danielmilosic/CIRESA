
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import os
import pandas as pd
from tqdm.notebook import tqdm
from CIRESA.utils import suppress_output,spacecraft_ID, pad_data_with_nans
from CIRESA import propagation, get_coordinates
from CIRESA.plot_heliosphere import plot_CIR_carrington, plot_spacecraft_carrington, make_movie
from sunpy.coordinates import get_horizons_coord
pd.options.mode.chained_assignment = None  # defvirtual_spacecraft_dflt='warn'

def Model_run(df_list, directory='Modelmovie'
                            , persistance=10, plot_cadence=24, sim_resolution = 0.5
                            , model='ballistic'
                            , CIR=False
                            , HEE = False
                            , back_prop = False
                            , movie=False
                            , rlim = 1.2
                            , COR = 0
                            , virtual_spacecraft = None # def virtual_spacecraft_df is Earth
                            , return_model = False
                            , save_plots = True
                            , pad = False
                            , framerate = 10
                            , variable_to_plot = 'V'
                            ):
    
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
        pad (bool): Whether to pad the data, so that the movie starts empty and gradually fills up
    """

    matplotlib.use('Agg')  # Non-GUI backend for headless plot generation
    
    #SORT

    if not isinstance(df_list, list):
        df_list = [df_list]
    last_values = [df['CARR_LON_RAD'].iloc[-1] for df in df_list]
    value_df_pairs = list(zip(last_values, df_list))
    sorted_value_df_pairs = sorted(value_df_pairs, key=lambda x: x[0])
    df_list = [df for _, df in sorted_value_df_pairs]

    if pad:
        # Pad each DataFrame in df_list
        padded_df_list = []
        for df in df_list:

            padded_df = pad_data_with_nans(df
                                           , before=df.index.min() - pd.Timedelta(days=persistance)
                                           , after=df.index.max() + pd.Timedelta(days=persistance))
            
            padded_df_list.append(padded_df)

        # Concatenate all padded DataFrames
        concat_df = pd.concat(padded_df_list)

    else:
        concat_df = pd.concat(df_list)

    # Create directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Calculate total time range in hours
    timerange = [concat_df.index.min(), concat_df.index.max()]
    total_hours = (timerange[1] - timerange[0]).total_seconds() / 3600

    # Number of steps based on the specified hour interval
    num_steps = int(total_hours // plot_cadence)

    P2D = pd.DataFrame()
    # Ensure that we're not exceeding the available range
    for i in tqdm(range(num_steps - (persistance * 24) // plot_cadence + 1)):
        #print(f'Timestep {i} out of {num_steps - (persistance * 24) // plot_cadence}')

        # Define the time window for the current plot (lower and upper index)
        lower_index = concat_df.index[0] + pd.Timedelta(hours=i * plot_cadence)
        upper_index = lower_index + pd.Timedelta(days=persistance)

        #SORT AGAIN, FIND Spacecraft positions

        last_values = []
        spacecraft_IDs = []

        carr = suppress_output(get_coordinates.get_carrington_longitude, upper_index)
        Earth_inert = suppress_output(get_horizons_coord, '3', upper_index)
        stereo_a_inert = suppress_output(get_horizons_coord, 'STEREO-A', upper_index)
        if upper_index > pd.to_datetime('2020-FEB-10 05:00:00'):
            solo_inert = suppress_output(get_horizons_coord, 'Solar Orbiter', upper_index)
        if upper_index > pd.to_datetime('2018-NOV-01 00:00:00'):
            psp_inert = suppress_output(get_horizons_coord, 'PSP', upper_index)
        maven_inert = suppress_output(get_horizons_coord, 'MAVEN', upper_index)

        for df in df_list:

            if upper_index in df['CARR_LON_RAD'].index:
                if isinstance(df['CARR_LON_RAD'].loc[upper_index], pd.Series):
                    last_longitude = df['CARR_LON_RAD'].loc[upper_index].iloc[0] * 360 / np.pi
                else:
                    last_longitude = df['CARR_LON_RAD'].loc[upper_index] * 360 / np.pi

            if 'Spacecraft_ID' in df.columns:
                spacecraft_IDs.append(spacecraft_ID(df))

                if spacecraft_ID(df)  == 'OMNI':
                    last_longitude = Earth_inert.lon.value
                elif spacecraft_ID(df)  == 'PSP':
                    if upper_index > pd.to_datetime('2018-NOV-01 00:00:00'):
                        last_longitude = psp_inert.lon.value# - Earth_inert.lon.value-carr[0]
                elif spacecraft_ID(df)  == 'SolO':
                    if upper_index > pd.to_datetime('2020-FEB-10 05:00:00'):
                        last_longitude = solo_inert.lon.value# - Earth_inert.lon.value-carr[0]
                elif spacecraft_ID(df)  == 'STEREO-A':
                    last_longitude = stereo_a_inert.lon.value# - Earth_inert.lon.value-carr[0]
                elif spacecraft_ID(df)  == 'MAVEN':
                    last_longitude = maven_inert.lon.value# - Earth_inert.lon.value-carr[0]

            else:
                last_longitude = np.nan
            
            last_values.append(last_longitude)
        
        last_values = np.array(last_values)+360

        value_df_pairs = list(zip(last_values, df_list, spacecraft_IDs))
        sorted_value_df_pairs = sorted(value_df_pairs, key=lambda x: x[0])
        df_list = [df for _, df, _ in sorted_value_df_pairs]
        spacecraft_IDs = [IDs for _, _, IDs in sorted_value_df_pairs]
        

        # solo_df = next((df for df, ID in zip(df_list, spacecraft_IDs) if ID == 'SolO'), None)
        # psp_df = next((df for df, ID in zip(df_list, spacecraft_IDs) if ID == 'PSP'), None)
        # stereo_a_df = next((df for df, ID in zip(df_list, spacecraft_IDs) if ID == 'STEREO-A'), None)
        # omni_df = next((df for df, ID in zip(df_list, spacecraft_IDs) if ID == 'OMNI'), None)
        # maven_df = next((df for df, ID in zip(df_list, spacecraft_IDs) if ID == 'MAVEN'), None)

    
        filtered_df_list = []
        for j, df in enumerate(df_list):
            if not df.empty:
                try:
                    filtered_df = df[(df.index >= lower_index) & (df.index <= upper_index)]
                    filtered_df_list.append(pad_data_with_nans(filtered_df, lower_index, upper_index))
                except Exception as e:
                    print(f'Error at {upper_index}: Failed to pad data with NaNs for spacecraft {spacecraft_ID(df)}')
        # Concatenate all filtered DataFrames
        insitu_df_slice = pd.concat(filtered_df_list)

        # Initialize a list to store the simulation DataFrames
        sim_df = []
        

        ### MODELRUNS

        # Iterate over the provided DataFrames and simulate using the selected model
        for df in filtered_df_list:
            df_slice = df#[(df.index >= lower_index) & (df.index <= upper_index)]
            if len(df.dropna(subset=['V', 'N']))>1:
                if not df_slice.empty:
                    # Apply the chosen propagation model
                    if model == 'inelastic':
                        #print('MODELRUN')
                        sim = propagation.inelastic_radial(df_slice, degree_resolution = sim_resolution, COR=COR)
                    elif model == 'ballistic':
                        #sim = suppress_output(propagation.ballistic, df_slice, degree_resolution = sim_resolution)
                        #print(df_slice['R'])
                        sim = propagation.ballistic( df_slice, degree_resolution = sim_resolution)

                    else:
                        #print('Unsupported model:', model)
                        sim  = df*np.nan
                    if back_prop:
                        sim_back = suppress_output(propagation.ballistic_reverse,df_slice, degree_resolution = sim_resolution)
                        sim = pd.concat([sim_back, sim])
                    
                    sim_df.append(sim)


        # Concatenate the in-situ data and simulation results for plotting
        insitu_df_slice['ITERATION']=0
        plot_df = pd.concat([insitu_df_slice] + sim_df)
        analyze_df = sim_df

        # CUT OUT THE TIME SERIES

        if model == 'inelastic' or model == 'ballistic':
            virtual_spacecraft_df = [] 

            for df in analyze_df:

                # Check if virtual_spacecraft is a string and filter the DataFrame
                if isinstance(virtual_spacecraft, str):
                    for df_virtual in filtered_df_list:
                        if 'Spacecraft_ID' in df_virtual.columns:  # Ensure the column exists to avoid errors
                            if spacecraft_ID(df_virtual) == virtual_spacecraft :
                                df = propagation.cut_from_sim(df.dropna(subset=['V']), df_virtual)
                                
                else:
                    try:
                        df = propagation.cut_from_sim(df, virtual_spacecraft)
                    except Exception:
                        print(f'Error: Could not cut from simulation for spacecraft {spacecraft_ID(df)}')
                        continue
                
                # Append the processed DataFrame to the list
                virtual_spacecraft_df.append(df)

            virtual_spacecraft_df = pd.concat(virtual_spacecraft_df)
            virtual_spacecraft_df['CARR_LON'] = (virtual_spacecraft_df['CARR_LON_RAD'] * 180/np.pi)%360
        
        else:
            virtual_spacecraft_df = insitu_df_slice
        
        ###PLOT

        if save_plots: 

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 14), subplot_kw={'projection': 'polar'})
            ax.remove()
            axes = fig.add_axes([0.1, 0.22, 0.8, 0.8], projection='polar')
            
            if HEE:
                plot_df['CARR_LON_RAD'] = plot_df['CARR_LON_RAD'] - carr/180*np.pi# + Earth_inert.heliocentricHEE.lon.value/180*np.pi
                axes.spines['polar'].set_visible(True)

            # Call the appropriate plot function (CIR or spacecraft Carrington plot)
            if CIR:
                plot_CIR_carrington(plot_df, rlim=rlim, axes=axes, s=30*sim_resolution)
            else:
                plot_spacecraft_carrington(plot_df, rlim=rlim, axes=axes, s=30*sim_resolution, variable_to_plot=variable_to_plot)
            
            if HEE:
                if 'Spacecraft_ID' in plot_df.columns:
                    if (plot_df['Spacecraft_ID']==6).any() > 0:
                        sns.scatterplot(x= [0], y = [1], ax = axes, s=100, color='blue', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==1).any() > 0:
                        sns.scatterplot(x= [(psp_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [psp_inert.radius.value], ax = axes, s=100, color='red', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==2).any() > 0:
                        sns.scatterplot(x= [(solo_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [solo_inert.radius.value], ax = axes, s=100, color='yellow', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==4).any() > 0:
                        sns.scatterplot(x= [(stereo_a_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [stereo_a_inert.radius.value], ax = axes, s=100, color='black', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==7).any() > 0:
                        sns.scatterplot(x= [(maven_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [maven_inert.radius.value], ax = axes, s=100, color='darkred', linewidth=0, legend=False)
                    
                else:
                    sns.scatterplot(x= [0], y = [1], ax = axes, s=100, color='blue', linewidth=0, legend=False)
                    sns.scatterplot(x= [(psp_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [psp_inert.radius.value], ax = axes, s=100, color='red', linewidth=0, legend=False)
                    sns.scatterplot(x= [(solo_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [solo_inert.radius.value], ax = axes, s=100, color='yellow', linewidth=0, legend=False)
                    sns.scatterplot(x= [(stereo_a_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [stereo_a_inert.radius.value], ax = axes, s=100, color='black', linewidth=0, legend=False)
                    sns.scatterplot(x= [(maven_inert.lon.value - Earth_inert.lon.value)/180*np.pi], y = [maven_inert.radius.value], ax = axes, s=100, color='darkred', linewidth=0, legend=False)
                
            else:
                if 'Spacecraft_ID' in plot_df.columns:
                    if (plot_df['Spacecraft_ID']==6).any() > 0:
                        sns.scatterplot(x= carr/180*np.pi, y = [1], ax = axes, s=50, color='blue', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==1).any() > 0:
                        sns.scatterplot(x= (psp_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [psp_inert.radius.value], ax = axes, s=50, color='red', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==2).any() > 0:
                        sns.scatterplot(x= (solo_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [solo_inert.radius.value], ax = axes, s=50, color='yellow', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==4).any() > 0:
                        sns.scatterplot(x= (stereo_a_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [stereo_a_inert.radius.value], ax = axes, s=50, color='black', linewidth=0, legend=False)
                    if (plot_df['Spacecraft_ID']==7).any() > 0:
                        sns.scatterplot(x= (maven_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [maven_inert.radius.value], ax = axes, s=50, color='darkred', linewidth=0, legend=False)
                
                else:
                    sns.scatterplot(x= carr/180*np.pi, y = [1], ax = axes, s=50, color='blue', linewidth=0, legend=False)
                    sns.scatterplot(x= (psp_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [psp_inert.radius.value], ax = axes, s=50, color='red', linewidth=0, legend=False)
                    sns.scatterplot(x= (solo_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [solo_inert.radius.value], ax = axes, s=50, color='yellow', linewidth=0, legend=False)
                    sns.scatterplot(x= (stereo_a_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [stereo_a_inert.radius.value], ax = axes, s=50, color='black', linewidth=0, legend=False)
                    sns.scatterplot(x= (maven_inert.lon.value - Earth_inert.lon.value + carr)/180*np.pi, y = [maven_inert.radius.value], ax = axes, s=50, color='darkred', linewidth=0, legend=False)

            ### PLOT THE TIME SERIES
            # Add a second set of normal (Cartesian) axes below the polar plot
            ax_timeseries = fig.add_axes([0.1, 0.05, 0.8, 0.2])  # [left, bottom, width, height]

            
            if HEE:
                virtual_spacecraft_df['CARR_LON_RAD'] = (virtual_spacecraft_df['CARR_LON_RAD'] - carr/180*np.pi)# + Earth_inert.heliocentricHEE.lon.value/180*np.pi
                virtual_spacecraft_df['CARR_LON'] = (virtual_spacecraft_df['CARR_LON_RAD'] * 180/np.pi +180)%360 - 180
                #virtual_spacecraft_df['CARR_LON'] = np.where(virtual_spacecraft_df['CARR_LON'] > 180, virtual_spacecraft_df['CARR_LON'] - 360, virtual_spacecraft_df['CARR_LON'])
                
            if CIR:
                #virtual_spacecraft_df['CARR_LON'] = virtual_spacecraft_df['CARR_LON_RAD']
                if len(virtual_spacecraft_df[virtual_spacecraft_df['Region']==1])>0:
                    sns.scatterplot(data=virtual_spacecraft_df[virtual_spacecraft_df['Region']==1], x='CARR_LON', y = variable_to_plot, ax = ax_timeseries, s=5, color='orange', linewidth=0, legend=False)

                if len(virtual_spacecraft_df[virtual_spacecraft_df['Region']==2])>0:
                    sns.scatterplot(data=virtual_spacecraft_df[virtual_spacecraft_df['Region']==2], x='CARR_LON', y = variable_to_plot, ax = ax_timeseries, s=5, color='red', linewidth=0, legend=False)
                    
                if len(virtual_spacecraft_df[virtual_spacecraft_df['Region']==3])>0: 
                    sns.scatterplot(data=virtual_spacecraft_df[virtual_spacecraft_df['Region']==3], x='CARR_LON', y = variable_to_plot, ax = ax_timeseries, s=5, color='black', linewidth=0, legend=False)
        
            else:  
                custom_palette = {
                        6: 'blue',
                        7: 'darkred',
                        2: 'orange',
                        4: 'black',
                        1: 'red',
                    }     
                sns.scatterplot(data=virtual_spacecraft_df, x='CARR_LON', y = variable_to_plot, ax = ax_timeseries, s=5, hue = virtual_spacecraft_df['Spacecraft_ID'], palette=custom_palette, linewidth=0, legend=False)

            if variable_to_plot =='V':
                ymin = 300
                ymax = 800
            else:
                ymin=np.min(virtual_spacecraft_df[variable_to_plot])
                ymax=np.max(virtual_spacecraft_df[variable_to_plot])
                #print(ymin, ymax)

            if not HEE:
                if 'Spacecraft_ID' in plot_df.columns:
                    ax_timeseries.set_xlim(360,0)
                    if (plot_df['Spacecraft_ID']==6).any() > 0:
                        ax_timeseries.vlines(x = carr, ymin=ymin, ymax=ymax, color='blue', label='Earth')
                    if (plot_df['Spacecraft_ID']==1).any() > 0:
                        ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='red', label='PSP')
                    if (plot_df['Spacecraft_ID']==2).any() > 0:
                        ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='orange', label='SolO')
                    if (plot_df['Spacecraft_ID']==4).any() > 0:
                        ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='black', label='STEREO-A')
                    if (plot_df['Spacecraft_ID']==7).any() > 0:  
                        ax_timeseries.vlines(x = maven_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='darkred', label='MAVEN')
                    ax_timeseries.set_xlabel('CARR_LON')
                else: 
                    ax_timeseries.set_xlim(360,0)
                    ax_timeseries.vlines(x = carr, ymin=ymin, ymax=ymax, color='blue', label='Earth')
                    ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='red', label='PSP')
                    ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='orange', label='SolO')
                    ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='black', label='STEREO-A')
                    ax_timeseries.vlines(x = maven_inert.lon.value - Earth_inert.lon.value + carr, ymin=ymin, ymax=ymax, color='darkred', label='MAVEN')
                    ax_timeseries.set_xlabel('CARR_LON')                
            else: 
                if 'Spacecraft_ID' in plot_df.columns:
                    ax_timeseries.set_xlabel('HEE_LON')
                    ax_timeseries.set_xlim(180,-180)
                    if (plot_df['Spacecraft_ID']==6).any() > 0:
                        ax_timeseries.vlines(x = 0, ymin=ymin, ymax=ymax, color='blue', label='Earth')
                    if (plot_df['Spacecraft_ID']==1).any() > 0:    
                        ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='red', label='PSP')
                    if (plot_df['Spacecraft_ID']==2).any() > 0:    
                        ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='orange', label='SolO')
                    if (plot_df['Spacecraft_ID']==4).any() > 0:    
                        ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='black', label='STEREO_A')
                    if (plot_df['Spacecraft_ID']==7).any() > 0:    
                        ax_timeseries.vlines(x = maven_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='darkred', label='MAVEN')
                else:
                    ax_timeseries.set_xlabel('HEE_LON')
                    ax_timeseries.set_xlim(180,-180)
                    ax_timeseries.vlines(x = 0, ymin=ymin, ymax=ymax, color='blue', label='Earth')
                    ax_timeseries.vlines(x = psp_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='red', label='PSP')
                    ax_timeseries.vlines(x = solo_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='orange', label='SolO')
                    ax_timeseries.vlines(x = stereo_a_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='black', label='STEREO_A')
                    ax_timeseries.vlines(x = maven_inert.lon.value - Earth_inert.lon.value, ymin=ymin, ymax=ymax, color='darkred', label='MAVEN')
                                
            ax_timeseries.set_ylim(ymin,ymax)

            if not virtual_spacecraft:
                ax_timeseries.set_title('1 AU')
            else:
                if isinstance(virtual_spacecraft, (int, float)):
                    ax_timeseries.set_title(str(virtual_spacecraft)+' AU')
                else:
                    ax_timeseries.set_title(str(virtual_spacecraft))

            if not (model == 'inelastic' or model == 'ballistic'):
                ax_timeseries.set_title('In-situ Data')
            ax_timeseries.set_ylabel('km/s')
            ax_timeseries.legend(loc='upper right')

            fig.text(0.1, 0.27, upper_index.strftime('%Y-%m-%d %H:%M:%S'), fontsize=13)
            fig.text(0.05, 0.95, 'Model: ' + str(model), fontsize=15)
            fig.text(0.05, 0.93, 'Persistance: ' + str(persistance) + ' days', fontsize=15)

            # SAVE

            # Save the generated plot to a file
            filename = os.path.join(directory, 'plot_' + upper_index.strftime('%Y-%m-%d-%H-%M-%S') + '_.png')
            plt.savefig(filename, format='png')
            plt.close()  # Close the plot to free memory
            
        if return_model:
            
            #print(virtual_spacecraft_df)
            #virtual_spacecraft_df['DELAY'] = sim_resolution * virtual_spacecraft_df['ITERATION'] 
            virtual_spacecraft_df['Time'] = upper_index# - pd.to_timedelta(virtual_spacecraft_df['TT'], unit='hours')
            virtual_spacecraft_df.reset_index(drop=True)


            unique_spacecraft = virtual_spacecraft_df['Spacecraft_ID'].dropna().unique()

            for spacecraft in unique_spacecraft:
                
                try:
                    data_point = virtual_spacecraft_df[virtual_spacecraft_df['Spacecraft_ID']==spacecraft]
                    data_point = data_point.loc[data_point['TT'].idxmin()]

                    data_point_virtual_sc = virtual_spacecraft_df[virtual_spacecraft_df['Spacecraft_ID']
                                                                ==spacecraft_ID(virtual_spacecraft)]
                    data_point_virtual_sc = data_point_virtual_sc.loc[data_point_virtual_sc['TT'].idxmin()]

                    distance = data_point['CARR_LON'] - data_point_virtual_sc['CARR_LON']   
                    if abs(distance) < sim_resolution:#/180*np.pi:
                        
                        P2D = pd.concat([P2D, data_point], axis = 1)
                except Exception as e: 
                    print(f'Error at {i}: {e}')
                    continue
        
    if return_model:
        P2D = P2D.transpose()
        P2D.set_index('Time', inplace=True)
        return P2D

    # If movie flag is set, generate a movie from the saved plots
    if movie and save_plots:
        make_movie(directory, framerate=framerate, rename=True)
