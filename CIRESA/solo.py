def download(timeframe):
    import pyspedas

    pyspedas.solo.swa(trange=timeframe, time_clip=True, level='l2', datatype='pas-grnd-mom', get_support_data=True
                 , downloadonly=True)
    pyspedas.solo.swa(trange=timeframe, time_clip=False, level='l3', datatype='his-comp-10min', get_support_data=True
                  , downloadonly=True)
    pyspedas.solo.mag(trange=timeframe, time_clip=True, get_support_data=True
                 , downloadonly=True)
    
def reduce(timeframe, cadence):

    from CIRESA import filefinder, read_cdf_to_df, get_coordinates
    import pandas as pd
    import numpy as np
    from astropy.time import Time
    import os

    root_dir = 'C:/Users/14milosi/CIRESA/solar_orbiter_data/'
    
    dir_swa = root_dir + 'swa/science/l2/'
    dir_his = root_dir + 'swa/science/l3/'
    dir_mag = root_dir + 'mag/'

    swa_files = filefinder.find_files_in_timeframe(dir_swa, timeframe[0], timeframe[1])
    his_files = filefinder.find_files_in_timeframe(dir_his, timeframe[0], timeframe[1])
    mag_files = filefinder.find_files_in_timeframe(dir_mag, timeframe[0], timeframe[1])

    print(swa_files, his_files, mag_files)

    #MAG

    print('### EXTRACTING MAGNETIC FIELD DATA ###')

    if sum(os.path.getsize(f) for f in mag_files) > 3e8:
 
        print('### LARGE MAGNETIC FIELD FILES ###')
        mag_df = []

        for f in mag_files:
                
                print(os.path.getsize(f)/1000000, 'MB', f)
                mag_loop_df = read_cdf_to_df.read_cdf_files_to_dataframe([f], ['Epoch', 'B_RTN'])
                mag_loop_df['Time']= pd.to_datetime(Time(mag_loop_df['Epoch'], format='cdf_tt2000', scale='utc').iso)
                mag_loop_df['B_R'] = mag_loop_df['B_RTN'].apply(lambda lst: lst[0])
                mag_loop_df['B_T'] = mag_loop_df['B_RTN'].apply(lambda lst: lst[1])
                mag_loop_df['B_N'] = mag_loop_df['B_RTN'].apply(lambda lst: lst[2])
                
                mag_loop_df.drop('B_RTN', axis=1, inplace=True)
                mag_loop_df.set_index('Time', inplace=True)              
                mag_loop_df = mag_loop_df.resample(rule=cadence).median()
                mag_loop_df['B'] = np.sqrt(mag_loop_df['B_R']**2 + mag_loop_df['B_T']**2 + mag_loop_df['B_N']**2)
                
                mag_df.append(mag_loop_df)

        mag_df = pd.concat(mag_df, axis=0)

    else:

        mag_df = read_cdf_to_df.read_cdf_files_to_dataframe(mag_files, ['Epoch', 'B_RTN'])
        mag_df['Time']= pd.to_datetime(Time(mag_df['Epoch'], format='cdf_tt2000', scale='utc').iso)
        mag_df['B_R'] = mag_df['B_RTN'].apply(lambda lst: lst[0])
        mag_df['B_T'] = mag_df['B_RTN'].apply(lambda lst: lst[1])
        mag_df['B_N'] = mag_df['B_RTN'].apply(lambda lst: lst[2])
        mag_df['B'] = np.sqrt(mag_df['B_R']**2 + mag_df['B_T']**2 + mag_df['B_N']**2)
        
        mag_df.drop('B_RTN', axis=1, inplace=True)
        mag_df.set_index('Time', inplace=True)
        mag_df = mag_df.resample(rule=cadence).median()

    #SWA
    print('### EXTRACTING SWA DATA ###')

    swa_df = read_cdf_to_df.read_cdf_files_to_dataframe(swa_files, ['Epoch', 'V_RTN', 'N', 'P_RTN', 'T'])

    swa_df['V_R'] = swa_df['V_RTN'].apply(lambda lst: lst[0])
    swa_df['V_T'] = swa_df['V_RTN'].apply(lambda lst: lst[1])
    swa_df['V_N'] = swa_df['V_RTN'].apply(lambda lst: lst[2])
    swa_df['P_R'] = swa_df['P_RTN'].apply(lambda lst: lst[0])
    swa_df['P_T'] = swa_df['P_RTN'].apply(lambda lst: lst[1])
    swa_df['P_N'] = swa_df['P_RTN'].apply(lambda lst: lst[2])
    swa_df['T'] = swa_df['T']*11604.5
    swa_df['V'] = np.sqrt(swa_df['V_R']**2 + swa_df['V_T']**2 + swa_df['V_N']**2)
    swa_df['P'] = np.sqrt(swa_df['P_R']**2 + swa_df['P_T']**2 + swa_df['P_N']**2)
    swa_df['Time']= pd.to_datetime(Time(swa_df['Epoch'], format='cdf_tt2000', scale='utc').iso)

    swa_df.drop(columns='V_RTN', inplace=True)
    swa_df.drop(columns='P_RTN', inplace=True)
    swa_df.drop(columns='Epoch', inplace=True)

    swa_df.set_index('Time', inplace=True)
    swa_df_nod = swa_df[~swa_df.index.duplicated(keep='first')]
    swa_df = swa_df_nod.resample(cadence).median()

    # GET COORDINATES
    coord_df = swa_df.resample(rule='6H').median()
    carr_lons, solo_r, solo_lats, solo_lon = get_coordinates.get_coordinates(coord_df, 'Solar Orbiter')
    coord_df['CARR_LON'] = np.asarray(carr_lons) % 360
    coord_df['LAT'] = solo_lats
    solo_lon = np.asarray(solo_lon)
    if (solo_lon < -175).any() & (solo_lon > 175).any():
        solo_lon[solo_lon < 0] += 360

    coord_df['INERT_LON'] = solo_lon
    coord_df['R'] = solo_r

    coord_df = coord_df.reindex(swa_df.index).interpolate(method='linear')
    swa_df['CARR_LON'] = coord_df['CARR_LON'].copy()*np.nan
    swa_df.loc[coord_df.index, 'CARR_LON'] = get_coordinates.calculate_carrington_longitude_from_lon(coord_df.index, coord_df['INERT_LON'])
    swa_df['CARR_LON_RAD'] = swa_df['CARR_LON']/180*3.1415926
    swa_df['LAT'] = coord_df['LAT'].copy()
   
    swa_df['INERT_LON'] = coord_df['INERT_LON'].copy()
    swa_df['R'] = coord_df['R'].copy()

    #HIS
    print('### EXTRACTING HIS DATA ###')

    his_df = read_cdf_to_df.read_cdf_files_to_dataframe(his_files, ['EPOCH', 'O7_O6_RATIO', 'C6_C5_RATIO'])
    his_df['Time']= pd.to_datetime(Time(his_df['EPOCH'], format='cdf_tt2000', scale='utc').iso)

    his_df = his_df[his_df['O7_O6_RATIO'] > 0.]
    his_df.set_index('Time', inplace=True)
    his_df.drop(columns='EPOCH', inplace=True)
    his_df_nod = his_df[~his_df.index.duplicated(keep='first')]
    his_df = his_df_nod.resample(cadence).median()

    solo_df = pd.concat([his_df, swa_df, mag_df], axis=1)


    #Calculate further plasma parameters
    solo_df['P_t'] = (solo_df['N'] * solo_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    solo_df['P_B'] = solo_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    solo_df['P'] = solo_df['P_t'] + solo_df['P_B']
    solo_df['Beta'] = solo_df['P_t'] / solo_df['P_B']
    solo_df['POL'] = np.sign(solo_df['B_R'] - solo_df['B_T']*solo_df['R']*2.7*10**(-6)/solo_df['V'])
    solo_df['S_P'] = solo_df['T']/solo_df['N']**(2./3.)/11604.5

    return solo_df

def plot(solo_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns


    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)

    # Plot multiple time series using seaborn's lineplot in each subplot


    sns.lineplot(data=solo_df, x=solo_df.index, y='V', ax=axes[0], color='black')
    sns.lineplot(data=solo_df, x=solo_df.index, y='V_R', ax=axes[0], color='blue', alpha=0.5)
    axes[0].set_ylabel('V $[km s^{-1}]$')

    
    sns.lineplot(data=solo_df, x=solo_df.index, y='V_T', ax=axes[1], label='V_T')
    sns.lineplot(data=solo_df, x=solo_df.index, y='V_N', ax=axes[1], label='V_N')
    axes[1].set_ylabel('V_TN $[km s^{-1}]$')

    sns.lineplot(data=solo_df, x=solo_df.index, y='N', ax=axes[2], color='black')
    axes[2].set_ylabel('N $[cm^{-3}]$')


    sns.lineplot(data=solo_df, x=solo_df.index, y='P', ax=axes[3], color='black')
    axes[3].set_ylabel('P $[nPa]$')

    solo_df['polarity'] = ['+' if pol > 0 else '-' for pol in solo_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=solo_df, x=solo_df.index, y='B', ax=axes[4], hue='polarity', palette = colors, s=5, alpha=1)
    axes[4].set_ylabel('B $[nT]$')
    #axes[4].set_ylim([0, 20])

    sns.lineplot(data=solo_df, x=solo_df.index, y='B_R', ax=axes[5], color='red', label='B_R')
    sns.lineplot(data=solo_df, x=solo_df.index, y='B_T', ax=axes[5], color='green', label='B_T')
    sns.lineplot(data=solo_df, x=solo_df.index, y='B_N', ax=axes[5], color='blue', label='B_N')
    axes[5].set_ylabel('B $[nT]$')

    sns.lineplot(data=solo_df, x=solo_df.index, y='S_P', ax=axes[6], color='black')
    axes[6].fill_between(solo_df.index, 2.69, 4, color='grey', alpha=0.7)
    #axes[5].set_ylim([0, 50])
    axes[6].set_ylabel('$S_p$ $[eV cm^{2}]$')

    sns.lineplot(data=solo_df, x=solo_df.index, y='R', ax=axes[7], color='black')
    axes[7].set_ylabel('r $[AU]$')


    ax2 = axes[2].twinx()
    sns.lineplot(data=solo_df, x=solo_df.index, y='T',  ax=ax2, color='tab:blue')
    ax2.set_ylabel('T $[K]$')
    #ax2.set_ylim([0, 2000000])
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')


    ax5 = axes[7].twinx()
    sns.lineplot(data=solo_df, x=solo_df.index, y='LAT', ax=ax5, color='tab:blue')
    #ax5.set_ylim([-10, 50])
    ax5.set_ylabel('LAT $[°]$')
    ax5.spines['right'].set_color('tab:blue')
    ax5.yaxis.label.set_color('tab:blue')
    ax5.tick_params(axis='y', colors='tab:blue')

    ax3 = axes[3].twinx()
    sns.lineplot(data=solo_df, x=solo_df.index, y='Beta', ax=ax3, color='tab:blue')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_yscale('log')
    ax3.spines['right'].set_color('tab:blue')
    ax3.yaxis.label.set_color('tab:blue')
    ax3.tick_params(axis='y', colors='tab:blue')

    ax6 = axes[6].twinx()
    sns.scatterplot(data=solo_df, x=solo_df.index, y='O7_O6_RATIO', ax=ax6, s=5, color='tab:blue')
    sns.lineplot(data=solo_df, x=solo_df.index, y=0.145, ax=ax6, color='grey')
    ax6.set_ylim([0, 0.3])
    ax6.set_ylabel('$O^{7+}/O^{6+}$')
    ax6.spines['right'].set_color('tab:blue')
    ax6.yaxis.label.set_color('tab:blue')
    ax6.tick_params(axis='y', colors='tab:blue')


    # # Customize the x-axis locator and formatter to have one date label for each tick
    # #locator = AutoDateLocator()
    # locator = DayLocator()
    # formatter = DateFormatter("%y-%m-%d %H:%M")
    # axes[-1].xaxis.set_major_locator(locator)
    # axes[-1].xaxis.set_major_formatter(formatter)
    # plt.xticks(rotation=45)


    # #axes[0].set_title('solo')
    # axes[1].set_title('')
    # axes[2].set_title('')

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)



def load(month):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'C:/Users/14milosi/CIRESA/reduced_data/solo'

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
    root_dir = 'C:/Users/14milosi/CIRESA/solar_orbiter_data/'
    
    dir_swa = root_dir + 'swa/science/l2'
    dir_his = root_dir + 'swa/science/l3'
    dir_mag = root_dir + 'mag/'

    swa_files = filefinder.find_files_in_timeframe(dir_swa, timeframe[0], timeframe[1])
    his_files = filefinder.find_files_in_timeframe(dir_his, timeframe[0], timeframe[1])
    mag_files = filefinder.find_files_in_timeframe(dir_mag, timeframe[0], timeframe[1])

    # Combine all files to delete
    all_files = swa_files + his_files + mag_files

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

    from CIRESA import solo, filefinder
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if isinstance(month, str):
        month = [month]

    for m in month:
        if pd.to_datetime(m) < pd.to_datetime('2020-02') :
            print('### NO SOLO DATA BEFORE 2020-02 ###')
        else:

            if os.path.exists('reduced_data\solo\solo_data'+m+'.parquet'):
                solo_df = solo.load(m)

            else:
                timeframe = filefinder.get_month_dates(m)

                solo.download(timeframe)
                solo_df = solo.reduce(timeframe, cadence)
                solo_df.to_parquet('reduced_data\solo\solo_data'+m+'.parquet')

        try:
            # Plot and save the figure
            solo.plot(solo_df)
            plt.savefig(f'solar_orbiter_data/monthly_plots/solo_{m}.png')
            plt.close()  # Close the plot to free up memory
        except Exception as e:
            print(f"Error plotting data for {m}: {e}")
        finally:
            # Ensure solo.delete() is called regardless of success or failure
            solo.delete(m)
