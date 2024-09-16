import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta


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
    mag_url = 'https://spdf.gsfc.nasa.gov/pub/data/maven/mag/l2/sunstate-1sec/cdfs/'
    swia_url = 'https://spdf.gsfc.nasa.gov/pub/data/maven/swia/l2/onboardsvymom/'

    
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
        fetch_files_from_directory(mag_url, date, save_dir='maven_data/mag/')
        fetch_files_from_directory(swia_url, date, save_dir='maven_data/swia/')


def reduce(timeframe, cadence = '0.1H'):

    from CIRESA import filefinder, read_cdf_to_df, get_coordinates
    import pandas as pd
    import numpy as np
    from astropy.time import Time
    import os

    root_dir = 'maven_data/'
    
    dir_mag = root_dir + 'mag/'
    dir_swia = root_dir + 'swia/'

    mag_files = filefinder.find_files_in_timeframe(dir_mag, timeframe[0], timeframe[1])
    swia_files = filefinder.find_files_in_timeframe(dir_swia, timeframe[0], timeframe[1])
    print('mag files:', mag_files)
    print('### EXTRACTING MAGNETIC FIELD DATA ###')

    mag_df = read_cdf_to_df.read_cdf_files_to_dataframe(mag_files, ['epoch', 'OB_B', 'POSN'])
    
    mag_df['Time']= pd.to_datetime(Time(mag_df['epoch'], format='cdf_tt2000', scale='utc').iso)    
    mag_df['B_R'] = mag_df['OB_B'].apply(lambda lst: lst[0])
    mag_df['B_T'] = mag_df['OB_B'].apply(lambda lst: lst[1])
    mag_df['B_N'] = mag_df['OB_B'].apply(lambda lst: lst[2])
    mag_df['B'] = np.sqrt(mag_df['B_R']**2 + mag_df['B_T']**2 + mag_df['B_N']**2)
    mag_df['X'] = mag_df['POSN'].apply(lambda lst: lst[0])
    mag_df['Y'] = mag_df['POSN'].apply(lambda lst: lst[1])
    mag_df['Z'] = mag_df['POSN'].apply(lambda lst: lst[2])
    mag_df.drop(columns=['POSN', 'OB_B'], axis=1, inplace=True)
    mag_df.set_index('Time', inplace=True)
    mag_df = mag_df.resample(rule=cadence).median()


    print('### EXTRACTING SWIA DATA ###')

    swia_df = read_cdf_to_df.read_cdf_files_to_dataframe(swia_files, ['epoch', 'velocity_mso', 'density',
                                                                      'telem_mode', 'pressure', 'temperature_mso'])
    swia_df['V_R'] = -swia_df['velocity_mso'].apply(lambda lst: lst[0])
    swia_df['V_T'] = -swia_df['velocity_mso'].apply(lambda lst: lst[1])
    swia_df['V_N'] = -swia_df['velocity_mso'].apply(lambda lst: lst[2])
    swia_df['T'] = np.sqrt(swia_df['temperature_mso'].apply(lambda lst: lst[0])**2 
                           + swia_df['temperature_mso'].apply(lambda lst: lst[1])**2 
                           + swia_df['temperature_mso'].apply(lambda lst: lst[1]))*11604.5
    swia_df['V'] = np.sqrt(swia_df['V_R']**2 + swia_df['V_T']**2 + swia_df['V_N']**2)
    #swia_df['P'] = swia_df['pressure']
    swia_df['N'] = swia_df['density']
    swia_df['Time']= pd.to_datetime(Time(swia_df['epoch'], format='cdf_tt2000', scale='utc').iso)
    swia_df.set_index('Time', inplace=True)

    swia_df.drop(columns=['epoch', 'velocity_mso', 'density',
                          'pressure', 'temperature_mso'], axis=1,  inplace=True)

    swia_df = swia_df.resample(cadence).median()


    # GET COORDINATES
    coord_df = swia_df.resample(rule='6H').median()
    carr_lons, maven_r, maven_lats, maven_lon = get_coordinates.get_coordinates(coord_df, 'MAVEN')
    coord_df['CARR_LON'] = np.asarray(carr_lons) % 360
    coord_df['LAT'] = maven_lats

    maven_lon = np.asarray(maven_lon)
    if (maven_lon < -175).any() & (maven_lon > 175).any():
        maven_lon[maven_lon < 0] += 360

    coord_df['INERT_LON'] = maven_lon
    coord_df['R'] = maven_r

    coord_df = coord_df.reindex(swia_df.index).interpolate(method='linear')
    swia_df['CARR_LON'] = coord_df['CARR_LON'].copy()*np.nan
    swia_df.loc[coord_df.index, 'CARR_LON'] = get_coordinates.calculate_carrington_longitude_from_lon(coord_df.index, coord_df['INERT_LON'])
    swia_df['CARR_LON_RAD'] = swia_df['CARR_LON']/180*3.1415926
    swia_df['LAT'] = coord_df['LAT'].copy()
   
    swia_df['INERT_LON'] = coord_df['INERT_LON'].copy()
    swia_df['R'] = coord_df['R'].copy()

    maven_df = pd.concat([swia_df, mag_df], axis=1)

    #Calculate further plasma parameters
    maven_df['P_t'] = (maven_df['N'] * maven_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    maven_df['P_B'] = maven_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    maven_df['P'] = maven_df['P_t'] + maven_df['P_B']
    maven_df['Beta'] = maven_df['P_t'] / maven_df['P_B']
    maven_df['POL'] = np.sign(maven_df['B_R'] - maven_df['B_T']*maven_df['R']*2.7*10**(-6)/maven_df['V'])
    maven_df['S_P'] = maven_df['T']/maven_df['N']**(2./3.)/11604.5

    maven_df = maven_df[maven_df['telem_mode']< 0.5]
    
    return maven_df

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
    axes[6].set_ylim([0, 25])
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



def load(month):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'C:/Users/14milosi/CIRESA/reduced_data/maven'

    
    files = filefinder.find_parquet_files(root_dir, month)

    # Ensure 'files' is always a list, even if a single file path is returned
    if isinstance(files, str):
        files = [files]


    spacecraft = []
    for f in files:

        print(f)
        df = pd.read_parquet(f)
        spacecraft.append(df)

    return pd.concat(spacecraft)

def delete(month):
    
    from CIRESA import filefinder
    import os

    timeframe = filefinder.get_month_dates(month)

    root_dir = 'C:/Users/14milosi/CIRESA/maven_data/'
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