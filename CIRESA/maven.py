import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta
import numpy as np

def download_file(url, local_filename):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if the request was successful
    
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded: {local_filename}")

def fetch_files_from_directory(url, date_str, save_dir):
    """Recursively fetch files from the directory and subdirectories, and download them."""
    response = requests.get(url)
    
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for link in soup.find_all('a'):
        href = link.get('href')
        
        # Skip links that refer to parent directories or navigation
        if href in ['../', '/']:
            continue
        
        # Full URL of the current link
        full_url = url + href
        
        # Check if the link is a directory (ends with '/')
        if href.endswith('/'):
            # Determine if it's a year or month folder based on its length and format
            if len(href) == 5 and href[:4].isdigit():  # Year folder (e.g., '2017/')
                if href[:4] == date_str[:4]:
                    #print(f"Entering year folder: {full_url}")
                    fetch_files_from_directory(full_url, date_str, save_dir)
            elif len(href) == 3 and href[:2].isdigit():  # Month folder (e.g., '04/')
                # Check if the folder name matches the desired month
                if href[:2] == date_str[4:6]:
                    #print(f"Entering month folder: {full_url}")
                    fetch_files_from_directory(full_url, date_str, save_dir)
            else:
                if href in url:
                    continue
                #print(f"Entering folder: {full_url}")
                fetch_files_from_directory(full_url, date_str, save_dir)
        else:
            # Check if the file contains the date string
            if date_str in href:
                local_filename = os.path.join(save_dir, href)
                
                # Check if the file has already been downloaded
                if not os.path.exists(local_filename):
                    print(f"Downloading File: {full_url}")
                    download_file(full_url, local_filename)
                else:
                    print(f"File current: {local_filename}")


def download(timeframe):
    
    print('DOWNLOADING MAVEN DATA')
    maven_url = 'https://spdf.gsfc.nasa.gov/pub/data/maven/maven/l2/sunstate-1sec/cdfs/'
    maven_url = 'https://spdf.gsfc.nasa.gov/pub/data/maven/maven/l2/onboardsvymom/'
    insitu_url = 'https://spdf.gsfc.nasa.gov/pub/data/maven/insitu/kp-4sec/cdfs/'
    
    # Convert string dates to datetime objects
    start_date = datetime.strptime(timeframe[0], '%Y-%m-%d')
    end_date = datetime.strptime(timeframe[1], '%Y-%m-%d')
    
    # Initialize the list to hold dates
    date_list = []
    
    # Generate dates from start_date to end_date inclusive
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    
    for date in date_list:
        #fetch_files_from_directory(maven_url, date, save_dir='maven_data/maven/')
        #fetch_files_from_directory(maven_url, date, save_dir='maven_data/maven/')
        fetch_files_from_directory(insitu_url, date, save_dir='maven_data/maven/')

def reduce(timeframe, cadence = '0.1H'):

    from CIRESA import filefinder, read_cdf_to_df, get_coordinates
    from CIRESA.utils import suppress_output
    import pandas as pd
    import numpy as np
    from astropy.time import Time
    import os


    if isinstance(timeframe, str):
        timeframe = filefinder.get_month_dates(timeframe)

    root_dir = 'maven_data/'
    
    #dir_maven = root_dir + 'maven/'
    #dir_maven = root_dir + 'maven/'
    dir_maven = root_dir + 'maven/'

    #maven_files = filefinder.find_files_in_timeframe(dir_maven, timeframe[0], timeframe[1])
    #maven_files = filefinder.find_files_in_timeframe(dir_maven, timeframe[0], timeframe[1])
    maven_files = filefinder.find_files_in_timeframe(dir_maven, timeframe[0], timeframe[1])
    maven_files = [file for file in maven_files if os.path.getsize(file) >= 1_048_576]
    print('maven files:', maven_files)

           
    # GET COORDINATES

    print('DOWNLOADING COORDINATES')

    coords_df = pd.DataFrame()
    date_range = pd.date_range(timeframe[0], timeframe[1], freq='D').tolist()[:-1]
    
    for day in date_range:

        print(day)

        day_start = day
        day_end = day + pd.Timedelta(days=1)
        
        # Create coord_df for the day with 2-hour intervals
        coord_df = pd.DataFrame({'Time': pd.date_range(day_start, day_end, freq='2H')})
        coord_df.set_index('Time', inplace=True)

        # Get coordinates
        #print('between here')
        carr_lons, r, lats, lon = suppress_output(get_coordinates.get_coordinates, coord_df, 'MAVEN')
        #print('and here there is an erfa warning')
        coord_df['LAT'] = lats
        coord_df['R'] = r

        lon = np.asarray(lon)
        if (lon < -175).any() & (lon > 175).any():
            lon[lon < 0] += 360

        coord_df['INERT_LON'] = lon
        # Interpolate to match df's index

        coord_df = coord_df.resample(rule = cadence).interpolate(method='linear')
        carr_lon = get_coordinates.calculate_carrington_longitude_from_lon(
            coord_df.index, coord_df['INERT_LON']
        )
        
        coord_df['CARR_LON'] = carr_lon

        #print(coord_df)
        coords_df = pd.concat([coords_df, coord_df])

    coords_df = coords_df[~coords_df.index.duplicated(keep='first')]

    coords_df['CARR_LON_RAD'] = coords_df['CARR_LON'] / 180 * np.pi



    print('### EXTRACTING mavenNETIC FIELD DATA ###')
    
    if len(maven_files) > 0:
        maven_df = read_cdf_to_df.read_cdf_files_to_dataframe(maven_files
                                                              , ['epoch', 'MAG_field_MSO'
                                                                 , 'SPICE_spacecraft_MSO'
                                                                 , 'SWIA_Hplus_density'
                                                                 , 'SWIA_Hplus_flow_velocity_MSO'
                                                                 , 'SWIA_Hplus_temperature'
                                                                 , 'SWIA_dynamic_pressure'
                                                                 , 'NGIMS_CO2_density'
                                                                 , 'SWIA_Hplus_flow_velocity_MSO_data_quality'])
        
        maven_df['Time']= pd.to_datetime(Time(maven_df['epoch'], format='cdf_tt2000', scale='utc').iso)    
        maven_df['B_R'] = maven_df['MAG_field_MSO'].apply(lambda lst: lst[0])
        maven_df['B_T'] = maven_df['MAG_field_MSO'].apply(lambda lst: lst[1])
        maven_df['B_N'] = maven_df['MAG_field_MSO'].apply(lambda lst: lst[2])
        maven_df['B'] = np.sqrt(maven_df['B_R']**2 + maven_df['B_T']**2 + maven_df['B_N']**2)
        maven_df['X'] = maven_df['SPICE_spacecraft_MSO'].apply(lambda lst: lst[0])
        maven_df['Y'] = maven_df['SPICE_spacecraft_MSO'].apply(lambda lst: lst[1])
        maven_df['Z'] = maven_df['SPICE_spacecraft_MSO'].apply(lambda lst: lst[2])

        maven_df['V_R'] = -maven_df['SWIA_Hplus_flow_velocity_MSO'].apply(lambda lst: lst[0])
        maven_df['V_T'] = -maven_df['SWIA_Hplus_flow_velocity_MSO'].apply(lambda lst: lst[1])
        maven_df['V_N'] = -maven_df['SWIA_Hplus_flow_velocity_MSO'].apply(lambda lst: lst[2])
        maven_df['T'] = maven_df['SWIA_Hplus_temperature']*11604.5
        maven_df['V'] = np.sqrt(maven_df['V_R']**2 + maven_df['V_T']**2 + maven_df['V_N']**2)
        #maven_df['P'] = maven_df['pressure']
        maven_df['N'] = maven_df['SWIA_Hplus_density']
        maven_df['CO2'] = maven_df['NGIMS_CO2_density']
        maven_df['quality'] = maven_df['SWIA_Hplus_flow_velocity_MSO_data_quality'].apply(lambda lst: lst[0])
        #print(maven_df['quality'])
        maven_df = maven_df[ maven_df['quality'] > 0.5 ]
        
        maven_df['Time']= pd.to_datetime(Time(maven_df['epoch'], format='cdf_epoch', scale='utc').iso)
        maven_df.set_index('Time', inplace=True)
        #print(maven_df)

        maven_df.drop(columns=['epoch', 'MAG_field_MSO'
                            , 'SWIA_Hplus_density'
                            , 'SWIA_Hplus_flow_velocity_MSO'
                            , 'SWIA_Hplus_temperature'
                            , 'SWIA_dynamic_pressure'
                            , 'NGIMS_CO2_density'
                            , 'SWIA_Hplus_flow_velocity_MSO_data_quality'
                            , 'SPICE_spacecraft_MSO']
                            , axis=1,  inplace=True)
        #print(maven_df)
        maven_df = maven_df.resample(cadence).median()

        maven_df = pd.concat([coords_df, maven_df], axis=1)


        #Calculate further plasma parameters
        maven_df['P_t'] = (maven_df['N'] * maven_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
        maven_df['P_B'] = maven_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
        maven_df['P'] = maven_df['P_t'] + maven_df['P_B']
        maven_df['Beta'] = maven_df['P_t'] / maven_df['P_B']
        maven_df['S_P'] = maven_df['T']/maven_df['N']**(2./3.)/11604.5
        maven_df['POL'] = np.sign(maven_df['B_R'] - maven_df['B_T']*maven_df['R']*2.7*10**(-6)/maven_df['V'])
        
        maven_df['Spacecraft_ID'] = 7
    
    else:
        print('### NO MAVEN FILES ###')
        index = pd.to_datetime([timeframe[0]])
        maven_df = pd.DataFrame(index=index,
                            columns=['V_R', 'V_T', 'V_N'
                                     , 'V', 'T', 'R', 'N'])
        
        maven_df = pd.concat([coords_df, maven_df], axis=1)
        maven_df['Spacecraft_ID'] = 7
        
    return maven_df

import pandas as pd
import numpy as np


def filter_sw(df, cadence='6H', interpolate=False):
    
    def process_chunk(chunk):
        if chunk.empty:
            return pd.DataFrame()  # Return an empty DataFrame if the chunk is empty

        try:
            chunk = chunk[chunk['V'] < 1000]
            chunk = chunk[chunk['V'] > 250]
            chunk = chunk[chunk['V_T'] < 500]
            chunk = chunk[chunk['N'] < 150]
        except KeyError:
            pass

        try:
            chunk = chunk[chunk['P'] < 1000]
        except KeyError:
            pass

        try:
            chunk = chunk[chunk['B'] < 100]
        except KeyError:
            pass

        try:
            chunk = chunk[chunk['quality'] > 0.99]
        except KeyError:
            pass

        # Resampling logic
        maxi = chunk.resample(cadence).max()
        mini = chunk.resample(cadence).min()
        medi = chunk.resample(cadence).median()

        resampled_chunk = maxi
        try:
            resampled_chunk[['B_R', 'B_T', 'B_N', 'V_T', 'LAT', 'CARR_LON']] = medi[['B_R', 'B_T', 'B_N', 'V_T', 'LAT', 'CARR_LON']]
        except KeyError:
            pass

        try:
            resampled_chunk['P_t'] = (resampled_chunk['N'] * resampled_chunk['V']**2) / 10**19 / 1.6727 * 10**6 * 10**9  # J/cm^3 to nPa
            resampled_chunk['S_P'] = resampled_chunk['T'] / resampled_chunk['N']**(2./3.) / 11604.5
        except KeyError:
            pass

        try:
            resampled_chunk['P_B'] = resampled_chunk['B']**2 / 2. / 1.25663706212 * 10**(-6) / 10**9 * 10**6 * 10**9  # nT to T
            resampled_chunk['Beta'] = resampled_chunk['P_t'] / resampled_chunk['P_B']
            resampled_chunk['POL'] = np.sign(
                resampled_chunk['B_R'] - resampled_chunk['B_T'] * resampled_chunk['R'] * 2.7 * 10**(-6) / resampled_chunk['V']
            )
        except KeyError:
            pass

        if interpolate:
            try:
                # Ensure valid start and end times
                if pd.isna(chunk.index.min()) or pd.isna(chunk.index.max()):
                    return resampled_chunk  # Skip interpolation if timestamps are invalid

                new_time_index = pd.date_range(start=chunk.index.min(), end=chunk.index.max(), freq='0.1H')
                resampled_chunk = resampled_chunk.reindex(new_time_index)
                resampled_chunk = resampled_chunk.interpolate(method='linear')
                resampled_chunk[['LAT', 'CARR_LON', 'INERT_LON', 'CARR_LON_RAD']] = chunk[['LAT', 'CARR_LON', 'INERT_LON', 'CARR_LON_RAD']]
            except KeyError:
                pass

        return resampled_chunk

    # Split input data into monthly chunks
    monthly_chunks = [chunk for _, chunk in df.groupby(pd.Grouper(freq='M'))]

    # Filter out invalid chunks
    valid_chunks = [chunk for chunk in monthly_chunks if not chunk.empty and not pd.isna(chunk.index.min())]

    # Process each valid chunk and concatenate the results
    processed_chunks = [process_chunk(chunk) for chunk in valid_chunks]
    result = pd.concat(processed_chunks)
    old_df = df.loc[result.index]
    try:
        result[['LAT', 'CARR_LON', 'INERT_LON', 'CARR_LON_RAD']] = old_df[['LAT', 'CARR_LON', 'INERT_LON', 'CARR_LON_RAD']]
    except Exception as e:
        print(f'Error processing data: {e}')
    #print(len(result), len(df.loc[result.index]))
    return result

def plot(maven_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)

    # Plot multiple time series using seaborn's lineplot in each subplot


    sns.lineplot(data=maven_df, x=maven_df.index, y='V', ax=axes[0], color='black')
    #sns.lineplot(data=maven_df, x=maven_df.index, y='V_R', ax=axes[0], color='blue', alpha=0.5)
    axes[0].set_ylabel('V $[km s^{-1}]$')

    
    sns.lineplot(data=maven_df, x=maven_df.index, y='V_T', ax=axes[1], label='V_T')
    #sns.lineplot(data=maven_df, x=maven_df.index, y='V_N', ax=axes[1], label='V_N')
    axes[1].set_ylabel('V_T $[km s^{-1}]$')

    sns.lineplot(data=maven_df, x=maven_df.index, y='N', ax=axes[2], color='black')
    axes[2].set_ylabel('N $[cm^{-3}]$')


    sns.lineplot(data=maven_df, x=maven_df.index, y='P', ax=axes[3], color='black')
    axes[3].set_ylabel('P $[nPa]$')

    maven_df['polarity'] = ['+' if pol > 0 else '-' for pol in maven_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=maven_df, x=maven_df.index, y='B', ax=axes[4], hue='polarity', palette = colors, s=5, alpha=1)
    axes[4].set_ylabel('B $[nT]$')
    #axes[4].set_ylim([0, 20])

    sns.lineplot(data=maven_df, x=maven_df.index, y='B_R', ax=axes[5], color='red', label='B_R')
    sns.lineplot(data=maven_df, x=maven_df.index, y='B_T', ax=axes[5], color='green', label='B_T')
    sns.lineplot(data=maven_df, x=maven_df.index, y='B_N', ax=axes[5], color='blue', label='B_N')
    axes[5].set_ylabel('B $[nT]$')

    sns.lineplot(data=maven_df, x=maven_df.index, y='S_P', ax=axes[6], color='black')
    axes[6].fill_between(maven_df.index, 2.69, 4, color='grey', alpha=0.7)
    axes[6].set_ylim([0, 250])
    axes[6].set_ylabel('$S_p$ $[eV cm^{2}]$')

    sns.lineplot(data=maven_df, x=maven_df.index, y='R', ax=axes[7], color='black')
    axes[7].set_ylabel('r $[AU]$')


    ax2 = axes[2].twinx()
    sns.lineplot(data=maven_df, x=maven_df.index, y='T',  ax=ax2, color='tab:blue')
    ax2.set_ylabel('T $[K]$')
    #ax2.set_ylim([0, 2000000])
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')


    ax5 = axes[7].twinx()
    sns.lineplot(data=maven_df, x=maven_df.index, y='LAT', ax=ax5, color='tab:blue')
    #ax5.set_ylim([-10, 50])
    ax5.set_ylabel('LAT $[°]$')
    ax5.spines['right'].set_color('tab:blue')
    ax5.yaxis.label.set_color('tab:blue')
    ax5.tick_params(axis='y', colors='tab:blue')

    ax3 = axes[3].twinx()
    sns.lineplot(data=maven_df, x=maven_df.index, y='Beta', ax=ax3, color='tab:blue')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_yscale('log')
    ax3.spines['right'].set_color('tab:blue')
    ax3.yaxis.label.set_color('tab:blue')
    ax3.tick_params(axis='y', colors='tab:blue')


    # # Customize the x-axis locator and formatter to have one date label for each tick
    # #locator = AutoDateLocator()
    # locator = DayLocator()
    # formatter = DateFormatter("%y-%m-%d %H:%M")
    # axes[-1].xaxis.set_major_locator(locator)
    # axes[-1].xaxis.set_major_formatter(formatter)
    # plt.xticks(rotation=45)


    # #axes[0].set_title('maven')
    # axes[1].set_title('')
    # axes[2].set_title('')

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)



def load(month='all', reduce_cadence=None, interpolate=False):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'reduced_data/maven'

    
    files = filefinder.find_parquet_files(root_dir, month)

    # Ensure 'files' is always a list, even if a single file path is returned
    if isinstance(files, str):
        files = [files]


    spacecraft = []
    for f in files:

        print(f)
        df = pd.read_parquet(f)

        if reduce_cadence is not None:
            df = filter_sw(df, reduce_cadence, interpolate=interpolate)

        spacecraft.append(df)
        result = pd.concat(spacecraft)
        result = result[~result.index.duplicated(keep='first')]

    return result

def delete(month):
    
    from CIRESA import filefinder
    import os

    timeframe = filefinder.get_month_dates(month)

    root_dir = 'maven_data/'
    maven_files = filefinder.find_files_in_timeframe(root_dir, timeframe[0], timeframe[1])

    # Print the files to be deleted
    print('Deleting:', maven_files)

    # Delete the files
    for file_path in maven_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete the directory and its contents
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def download_reduce_save_space(month, cadence):

    from CIRESA import maven, filefinder
    import os
    import matplotlib.pyplot as plt

    
    if isinstance(month, str):
        month = [month]

    for m in month:
        if pd.to_datetime(m) < pd.to_datetime('2014-03') :
            print('### NO MAVEN DATA BEFORE 2018-10 ###')
        else:
            if os.path.exists('reduced_data\maven\maven_data'+m+'.parquet'):
                maven_df = maven.load(m)

            else:
                timeframe = filefinder.get_month_dates(m)

                maven.download(timeframe)
                maven_df = maven.reduce(timeframe, cadence)
                maven_df.to_parquet('reduced_data\maven\maven_data'+m+'.parquet')

            try:
                # Plot and save the figure
                maven.plot(maven_df)
                plt.savefig(f'maven_data/monthly_plots/maven_{m}.png')
                plt.close()  # Close the plot to free up memory
            except Exception as e:
                print(f"Error plotting data for {m}: {e}")
            finally:
                # Ensure maven.delete() is called regardless of success or failure
                maven.delete(m)