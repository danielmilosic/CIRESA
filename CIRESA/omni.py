def download_omni(timeframe):
    import pyspedas

    pyspedas.omni.data(trange=timeframe, time_clip=True, datatype='1min', get_support_data=True
                , downloadonly=True)

    
def reduce_omni(timeframe, cadence):

    from CIRESA import filefinder, read_cdf_to_df, get_coordinates
    import pandas as pd
    import numpy as np
    from astropy.time import Time

    root_dir = 'C:/Users/14milosi/CIRESA/omni_data/'
    omni_files = filefinder.find_files_in_timeframe(root_dir, timeframe[0], timeframe[1])
    print(omni_files)

    variables_to_read = ['Epoch', 'proton_density', 
                         'Vy', 'flow_speed', 
                         'BX_GSE', 'BY_GSE', 'BZ_GSE',
                         'Pressure', 'T', 'Beta', 'IMF']
    variables_to_delete = ['Epoch', 'proton_density', 
                         'Vy', 'flow_speed', 
                         'BX_GSE', 'BY_GSE', 'BZ_GSE',
                         'Pressure', 'IMF']
    #variables_to_read = ['Epoch', 'proton_density', 'Vy', 'flow_speed', 'pressure', 'T']
    omni_df = read_cdf_to_df.read_cdf_files_to_dataframe(omni_files, variables_to_read)
    Time(omni_df['Epoch'], format='cdf_epoch', scale='utc').datetime
    omni_df['Time']= pd.to_datetime(Time(omni_df['Epoch'], format='cdf_epoch', scale='utc').iso)
    omni_df.set_index('Time', inplace=True)
    omni_df = omni_df[(omni_df['flow_speed'] > 0) & (omni_df['flow_speed'] < 1000)]
    omni_df['N'] = omni_df['proton_density']
    omni_df['V'] = omni_df['flow_speed']
    omni_df['V_T'] = omni_df['Vy']
    omni_df['B_R'] = omni_df['BX_GSE']
    omni_df['B_T'] = omni_df['BY_GSE']
    omni_df['B_N'] = omni_df['BZ_GSE']
    omni_df['B'] = np.sqrt(omni_df['B_R']**2 + omni_df['B_T']**2 + omni_df['B_N']**2)
    omni_df = omni_df[(omni_df['B_N'] < 100)]
    omni_df['P'] = omni_df['Pressure']
    omni_df = omni_df.resample(rule=cadence).median()
    omni_df.drop(columns=variables_to_delete, axis=1, inplace=True)

 
    # GET COORDINATES
    coord_df = omni_df.resample(rule='6H').median()
    carr_lons, omni_r, omni_lats, omni_lon = get_coordinates.get_coordinates(coord_df, '3')
    coord_df['CARR_LON'] = carr_lons
    coord_df['LAT'] = omni_lats
    coord_df['INERT_LON'] = omni_lon
    coord_df['R'] = omni_r

    coord_df = coord_df.resample(rule=cadence).interpolate(method='linear')
    omni_df['CARR_LON'] = coord_df['CARR_LON'].copy()
    omni_df['CARR_LON_RAD'] = coord_df['CARR_LON']/180*3.1415926
    omni_df['LAT'] = coord_df['LAT'].copy()
    omni_df['INERT_LON'] = coord_df['INERT_LON'].copy()
    omni_df['R'] = coord_df['R'].copy()

    #Calculate further plasma parameters
    omni_df['P_t'] = (omni_df['N'] * omni_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    omni_df['P_B'] = omni_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    #omni_df['P'] = omni_df['P_t'] + omni_df['P_B']
    #omni_df['Beta'] = omni_df['P_t'] / omni_df['P_B']
    omni_df['POL'] = np.sign(omni_df['B_R'] - omni_df['B_T']*omni_df['R']*2.7*10**(-6)/omni_df['V'])
    omni_df['S_P'] = omni_df['T']/omni_df['N']**(2./3.)/11604.5

    return omni_df


def plot_omni(omni_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)

    # Plot multiple time series using seaborn's lineplot in each subplot


    sns.lineplot(data=omni_df, x=omni_df.index, y='V', ax=axes[0], color='black')
    #sns.lineplot(data=omni_df, x=omni_df.index, y='V_R', ax=axes[0], color='blue', alpha=0.5)
    axes[0].set_ylabel('V $[km s^{-1}]$')

    
    sns.lineplot(data=omni_df, x=omni_df.index, y='V_T', ax=axes[1], label='V_T')
    #sns.lineplot(data=omni_df, x=omni_df.index, y='V_N', ax=axes[1], label='V_N')
    axes[1].set_ylabel('V_T $[km s^{-1}]$')

    sns.lineplot(data=omni_df, x=omni_df.index, y='N', ax=axes[2], color='black')
    axes[2].set_ylabel('N $[cm^{-3}]$')


    sns.lineplot(data=omni_df, x=omni_df.index, y='P', ax=axes[3], color='black')
    axes[3].set_ylabel('P $[nPa]$')

    omni_df['polarity'] = ['+' if pol > 0 else '-' for pol in omni_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=omni_df, x=omni_df.index, y='B', ax=axes[4], hue='polarity', palette = colors, s=5, alpha=1)
    axes[4].set_ylabel('B $[nT]$')
    #axes[4].set_ylim([0, 20])

    sns.lineplot(data=omni_df, x=omni_df.index, y='B_R', ax=axes[5], color='red', label='B_R')
    sns.lineplot(data=omni_df, x=omni_df.index, y='B_T', ax=axes[5], color='green', label='B_T')
    sns.lineplot(data=omni_df, x=omni_df.index, y='B_N', ax=axes[5], color='blue', label='B_N')
    axes[5].set_ylabel('B $[nT]$')

    sns.lineplot(data=omni_df, x=omni_df.index, y='S_P', ax=axes[6], color='black')
    axes[6].fill_between(omni_df.index, 2.69, 4, color='grey', alpha=0.7)
    axes[6].set_ylim([0, 25])
    axes[6].set_ylabel('$S_p$ $[eV cm^{2}]$')

    sns.lineplot(data=omni_df, x=omni_df.index, y='R', ax=axes[7], color='black')
    axes[7].set_ylabel('r $[AU]$')


    ax2 = axes[2].twinx()
    sns.lineplot(data=omni_df, x=omni_df.index, y='T',  ax=ax2, color='tab:blue')
    ax2.set_ylabel('T $[K]$')
    #ax2.set_ylim([0, 2000000])
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')


    ax5 = axes[7].twinx()
    sns.lineplot(data=omni_df, x=omni_df.index, y='LAT', ax=ax5, color='tab:blue')
    #ax5.set_ylim([-10, 50])
    ax5.set_ylabel('LAT $[°]$')
    ax5.spines['right'].set_color('tab:blue')
    ax5.yaxis.label.set_color('tab:blue')
    ax5.tick_params(axis='y', colors='tab:blue')

    ax3 = axes[3].twinx()
    sns.lineplot(data=omni_df, x=omni_df.index, y='Beta', ax=ax3, color='tab:blue')
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


    # #axes[0].set_title('omni')
    # axes[1].set_title('')
    # axes[2].set_title('')

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)


def load_omni(month):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'C:/Users/14milosi/CIRESA/reduced_data/omni'

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

def delete_omni(month):
    
    from CIRESA import filefinder
    import os

    timeframe = filefinder.get_month_dates(month)

    root_dir = 'C:/Users/14milosi/CIRESA/omni_data/'
    omni_files = filefinder.find_files_in_timeframe(root_dir, timeframe[0], timeframe[1])

    # Print the files to be deleted
    print('Deleting:', omni_files)

    # Delete the files
    for file_path in omni_files:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete the directory and its contents
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def download_reduce_save_space_omni(month, cadence):

    from CIRESA import omni, filefinder
    import os
    import matplotlib.pyplot as plt

    
    if isinstance(month, str):
        month = [month]

    for m in month:

        if os.path.exists('reduced_data\omni\omni_data'+m+'.parquet'):
            omni_df = omni.load_omni(m)

        else:
            timeframe = filefinder.get_month_dates(m)

            omni.download_omni(timeframe)
            omni_df = omni.reduce_omni(timeframe, cadence)
            omni_df.to_parquet('reduced_data\omni\omni_data'+m+'.parquet')

        omni.plot_omni(omni_df)
        plt.savefig('omni_data/monthly_plots/omni'+m+'.png')
        omni.delete_omni(m)

