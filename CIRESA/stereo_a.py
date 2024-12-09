def download(timeframe):
    import pyspedas

    pyspedas.stereo.mag(trange=timeframe, time_clip=True, get_support_data=True
                    , downloadonly=True)
    pyspedas.stereo.plastic(trange=timeframe, time_clip=True, get_support_data=True, level = 'l2', datatype='1min'
                    , downloadonly=True)
    
def reduce(timeframe, cadence):

    from CIRESA import filefinder, read_cdf_to_df, get_coordinates
    import pandas as pd
    import numpy as np
    from astropy.time import Time

    root_dir = 'C:/Users/14milosi/CIRESA/stereo_data/'
    
    dir_impact = root_dir + '/impact/level1/ahead'
    dir_plastic = root_dir + '/plastic/level2/Protons/Derived_from_1D_Maxwellian/ahead'

    impact_files = filefinder.find_files_in_timeframe(dir_impact, timeframe[0], timeframe[1])
    plastic_files = filefinder.find_files_in_timeframe(dir_plastic, timeframe[0], timeframe[1])

    print(impact_files, plastic_files)

    #IMPACT MAG

    mag_df = read_cdf_to_df.read_cdf_files_to_dataframe(impact_files, ['epoch', 'BFIELD'])
    mag_df['Time']= pd.to_datetime(Time(mag_df['epoch'], format='cdf_epoch', scale='utc').iso)
    mag_df.set_index('Time', inplace=True)
    mag_df['B_R'] = mag_df['BFIELD'].apply(lambda lst: lst[0])
    mag_df['B_T'] = mag_df['BFIELD'].apply(lambda lst: lst[1])
    mag_df['B_N'] = mag_df['BFIELD'].apply(lambda lst: lst[2])
    mag_df = mag_df[mag_df['BFIELD'].apply(lambda lst: lst[0]) > -1000.]
    mag_df['B'] = np.sqrt(mag_df['B_R']**2 + mag_df['B_T']**2 + mag_df['B_N']**2)

    mag_df.drop('BFIELD', axis=1, inplace=True)
    mag_df.drop('epoch', axis=1, inplace=True)

    mag_df = mag_df.resample(rule=cadence).median()

    #PLASTIC

    variables_to_read = ['epoch', 'proton_bulk_speed', 'proton_number_density', 'proton_temperature',
                         'spcrft_lon_carr', 'spcrft_lon_hci', 'spcrft_lat_hci', 'heliocentric_dist',
                         'proton_Vt_RTN', 'proton_Vr_RTN', 'proton_Vn_RTN']

    plastic_df = read_cdf_to_df.read_cdf_files_to_dataframe(plastic_files, variables_to_read)
    plastic_df['Time']= pd.to_datetime(Time(plastic_df['epoch'], format='cdf_epoch', scale='utc').iso)
    plastic_df.set_index('Time', inplace=True)
    plastic_df['T'] = plastic_df['proton_temperature']
    plastic_df['N'] = plastic_df['proton_number_density']
    plastic_df['V'] = plastic_df['proton_bulk_speed']
    plastic_df['V_R'] = plastic_df['proton_Vr_RTN']
    plastic_df['V_T'] = plastic_df['proton_Vt_RTN']
    plastic_df['V_N'] = plastic_df['proton_Vn_RTN']
    plastic_df['CARR_LON'] = plastic_df['spcrft_lon_carr']
    plastic_df['CARR_LON_RAD'] = plastic_df['CARR_LON']/180*3.1415926
    plastic_df['LAT'] = plastic_df['spcrft_lat_hci']
    plastic_df['INERT_LON'] = plastic_df['spcrft_lon_hci']
    plastic_df['R'] = plastic_df['heliocentric_dist']/ 149597870.7

    plastic_df['P'] = (plastic_df['proton_number_density'] 
                       * plastic_df['proton_bulk_speed']**2) / 10**19 / 1.6727
    
    plastic_df.drop(columns=variables_to_read, axis=1, inplace=True)

    plastic_df = plastic_df[plastic_df['V'] > 0]
    #plastic_df = plastic_df[plastic_df['V_T'] > -300]

    plastic_df = plastic_df.resample(rule=cadence).median()
    

    stereo_a_df = pd.concat([plastic_df, mag_df], axis=1)

    stereo_a_df['P_t'] = (stereo_a_df['N'] * stereo_a_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    stereo_a_df['P_B'] = stereo_a_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    stereo_a_df['P'] = stereo_a_df['P_t'] + stereo_a_df['P_B']
    stereo_a_df['Beta'] = stereo_a_df['P_t'] / stereo_a_df['P_B']
    stereo_a_df['POL'] = np.sign(stereo_a_df['B_R'] - stereo_a_df['B_T']*stereo_a_df['R']*2.7*10**(-6)/stereo_a_df['V'])
    stereo_a_df['S_P'] = stereo_a_df['T']/stereo_a_df['N']**(2./3.)/11604.5

    stereo_a_df['Spacecraft_ID'] = 4

    return stereo_a_df


def plot(stereo_a_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 12), sharex=True)

    # Plot each series with try-except to handle missing columns
    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='V', ax=axes[0], color='black')
    except KeyError:
        pass
    axes[0].set_ylabel('$[km s^{-1}]$')

    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='N', ax=axes[1], color='black')
    except KeyError:
        pass
    axes[1].set_ylabel('N $[cm^{-3}]$')

    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='P', ax=axes[2], color='black')
    except KeyError:
        pass
    axes[2].set_ylabel('P $[nPa]$')

    try:
        stereo_a_df['polarity'] = ['+' if pol > 0 else '-' for pol in stereo_a_df['POL']]
        colors = {'-': 'tab:blue', '+': 'tab:red'}
        sns.scatterplot(data=stereo_a_df, x=stereo_a_df.index, y='B', ax=axes[3], hue='polarity', palette=colors, s=5, alpha=1)
    except KeyError:
        pass
    axes[3].set_ylabel('B $[nT]$')

    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='B_R', ax=axes[4], color='red', label='B_R')
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='B_T', ax=axes[4], color='green', label='B_T')
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='B_N', ax=axes[4], color='blue', label='B_N')
    except KeyError:
        pass
    axes[4].set_ylabel('B $[nT]$')

    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='S_P', ax=axes[5], color='black')
    except KeyError:
        pass
    axes[5].fill_between(stereo_a_df.index, 2.69, 4, color='grey', alpha=0.7)
    axes[5].set_ylabel('$S_p$ $[eV cm^{2}]$')

    try:
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='R', ax=axes[6], color='black')
    except KeyError:
        pass
    axes[6].set_ylabel('r $[AU]$')
    axes[6].set_ylim([0.9, 1.1])
    
    # Handle twin axes with try-except
    try:
        ax2 = axes[1].twinx()
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='T', ax=ax2, color='tab:blue')
        ax2.set_ylabel('T $[K]$')
        ax2.spines['right'].set_color('tab:blue')
        ax2.yaxis.label.set_color('tab:blue')
        ax2.tick_params(axis='y', colors='tab:blue')
    except KeyError:
        pass

    try:
        ax5 = axes[6].twinx()
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='LAT', ax=ax5, color='tab:blue')
        ax5.set_ylabel('LAT $[°]$')
        ax5.spines['right'].set_color('tab:blue')
        ax5.yaxis.label.set_color('tab:blue')
        ax5.tick_params(axis='y', colors='tab:blue')
    except KeyError:
        pass

    try:
        ax3 = axes[2].twinx()
        sns.lineplot(data=stereo_a_df, x=stereo_a_df.index, y='Beta', ax=ax3, color='tab:blue')
        ax3.set_ylabel(r'$\beta$')
        ax3.set_yscale('log')
        ax3.spines['right'].set_color('tab:blue')
        ax3.yaxis.label.set_color('tab:blue')
        ax3.tick_params(axis='y', colors='tab:blue')
    except KeyError:
        pass

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=0.1)
    plt.show()


def load(month):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'C:/Users/14milosi/CIRESA/reduced_data/stereo_a'

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
    root_dir = 'C:/Users/14milosi/CIRESA/stereo_data/'
    
    dir_impact = root_dir + '/impact/level1/ahead'
    dir_plastic = root_dir + '/plastic/level2/Protons/Derived_from_1D_Maxwellian/ahead'

    impact_files = filefinder.find_files_in_timeframe(dir_impact, timeframe[0], timeframe[1])
    plastic_files = filefinder.find_files_in_timeframe(dir_plastic, timeframe[0], timeframe[1])

    print(impact_files, plastic_files)

    # Combine all files to delete
    all_files = impact_files + plastic_files

    # Print the files to be deleted
    print('Deleting:', all_files)

    # Delete the files
    for file_path in all_files:
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

    from CIRESA import stereo_a, filefinder
    import os
    import matplotlib.pyplot as plt

    
    if isinstance(month, str):
        month = [month]

    for m in month:

        if os.path.exists('reduced_data\stereo_a\stereo_a_data'+m+'.parquet'):
            stereo_a_df = stereo_a.load(m)

        else:
            timeframe = filefinder.get_month_dates(m)

            stereo_a.download(timeframe)
            stereo_a_df = stereo_a.reduce(timeframe, cadence)
            stereo_a_df.to_parquet('reduced_data\stereo_a\stereo_a_data'+m+'.parquet')

        try:
            # Plot and save the figure
            stereo_a.plot(stereo_a_df)
            plt.savefig(f'stereo_data/monthly_plots/stereo_a_{m}.png')
            plt.close()  # Close the plot to free up memory
        except Exception as e:
            print(f"Error plotting data for {m}: {e}")
        finally:
            # Ensure stereo_a.delete() is called regardless of success or failure
            stereo_a.delete(m)