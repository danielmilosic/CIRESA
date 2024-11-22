from CIRESA import filefinder, read_cdf_to_df, get_coordinates
import pandas as pd
import numpy as np
from astropy.time import Time
import os
from CIRESA.utils import suppress_output

def download(timeframe):
    import pyspedas

    print('### DOWNLOADING MAGNETIC FIELD DATA: 1 out 3 INSTRUMENTS ###')

    pyspedas.psp.fields(trange=timeframe, time_clip=True, datatype='mag_RTN', get_support_data=True
                    , downloadonly=True, level='l2')
    
    print('### DOWNLOADING SPI DATA: 2 out 3 INSTRUMENTS ###')
    
    pyspedas.psp.spi(trange=timeframe, time_clip=True, get_support_data=True
                 , downloadonly=True
                 , datatype='sf00_l3_mom'
                 )
    
    print('### DOWNLOADING SPC DATA: 3 out 3 INSTRUMENTS ###')
    
    pyspedas.psp.spc(trange=timeframe, downloadonly=True, datatype='l3i')

def reduce(timeframe, cadence='0.1H'):


    root_dir = 'C:/Users/14milosi/CIRESA/psp_data/'
    
    dir_fields = root_dir + 'fields/'
    dir_spi = root_dir + 'sweap/spi/'
    dir_spc = root_dir + 'sweap/spc/'

    fields_files = filefinder.find_files_in_timeframe(dir_fields, timeframe[0], timeframe[1])
    spi_files = filefinder.find_files_in_timeframe(dir_spi, timeframe[0], timeframe[1])
    spc_files = filefinder.find_files_in_timeframe(dir_spc, timeframe[0], timeframe[1])

    print(fields_files, spc_files, spi_files)


    #FIELDS

    print('### EXTRACTING MAGNETIC FIELD DATA ###')
    
    if len(fields_files) > 0:
        #if sum(os.path.getsize(f) for f in fields_files) > 3e8:

        print('### LARGE MAGNETIC FIELD FILES ###')
        fields_df = []

        for f in fields_files:
                
                print(os.path.getsize(f)/1000000, 'MB', f)

                fields_loop_df = read_cdf_to_df.read_cdf_files_to_dataframe([f], ['epoch_mag_RTN', 'psp_fld_l2_mag_RTN'])

                # Resample and drop unnecessary columns
                fields_loop_df = fields_loop_df.iloc[::10]
                fields_loop_df['Time'] = Time(fields_loop_df['epoch_mag_RTN'], format='cdf_tt2000', scale='utc').to_datetime()
                fields_loop_df.set_index('Time', inplace=True)
                fields_loop_df['B_R'] = fields_loop_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[0])
                fields_loop_df['B_T'] = fields_loop_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[1])
                fields_loop_df['B_N'] = fields_loop_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[2])
                fields_loop_df.drop('psp_fld_l2_mag_RTN', axis=1, inplace=True)
                fields_loop_df.drop('epoch_mag_RTN', axis=1, inplace=True)

                fields_loop_df = fields_loop_df.resample(rule=cadence).median(cadence)

                fields_loop_df['B'] = np.sqrt(fields_loop_df['B_R']**2 + fields_loop_df['B_T']**2 + fields_loop_df['B_N']**2)
                
                
                # GET COORDINATES
                coord_df = fields_loop_df.resample(rule='2H').median()
                carr_lons, psp_r, psp_lats, psp_lon = suppress_output(get_coordinates.get_coordinates, coord_df, 'PSP')
                
                coord_df['CARR_LON'] = carr_lons      
                coord_df['CARR_LON_RAD'] = coord_df['CARR_LON']/180*3.1415926
                coord_df['LAT'] = psp_lats

                psp_lon = np.asarray(psp_lon)
                if (psp_lon < -175).any() & (psp_lon > 175).any():
                    psp_lon[psp_lon < 0] += 360

                coord_df['INERT_LON'] = psp_lon

                coord_df = coord_df.reindex(fields_loop_df.index).interpolate(method='linear')
                fields_loop_df['CARR_LON'] = coord_df['CARR_LON'] *np.nan
                fields_loop_df.loc[coord_df.index, 'CARR_LON'] = get_coordinates.calculate_carrington_longitude_from_lon(coord_df.index, coord_df['INERT_LON'])
                fields_loop_df['CARR_LON_RAD'] = fields_loop_df['CARR_LON']/180*3.1415926
                fields_loop_df['LAT'] = coord_df['LAT'].copy()
                fields_loop_df['INERT_LON'] = coord_df['INERT_LON'].copy()
            

                fields_df.append(fields_loop_df)

        fields_df = pd.concat(fields_df, axis=0)

        # else:

        #     fields_df = read_cdf_to_df.read_cdf_files_to_dataframe(fields_files, ['epoch_mag_RTN', 'psp_fld_l2_mag_RTN'])

        #     # Resample and drop unnecessary columns
        #     fields_df = fields_df.iloc[::10]
        #     fields_df['Time'] = Time(fields_df['epoch_mag_RTN'], format='cdf_tt2000', scale='utc').to_datetime()
        #     fields_df.set_index('Time', inplace=True)
        #     fields_df['B_R'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[0])
        #     fields_df['B_T'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[1])
        #     fields_df['B_N'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[2])
        #     fields_df.drop('psp_fld_l2_mag_RTN', axis=1, inplace=True)
        #     fields_df.drop('epoch_mag_RTN', axis=1, inplace=True)

        #     fields_df = fields_df.resample(rule=cadence).median(cadence)

        #     fields_df['B'] = np.sqrt(fields_df['B_R']**2 + fields_df['B_T']**2 + fields_df['B_N']**2)
    
    else:  
        print('### NO MAGNETIC FIELDS FILES ###')
        index = pd.to_datetime([timeframe[0]])
        fields_df = pd.DataFrame(index=index,
                            columns=['B_R', 'B_T', 'B_N', 'B'])
       
    #SPI

    print('### EXTRACTING SPI DATA ###')

    if len(spi_files) > 0:

        spi_df = read_cdf_to_df.read_cdf_files_to_dataframe(spi_files, ['Time', 'VEL_RTN_SUN', 'DENS', 'TEMP', 'SUN_DIST'])

        spi_df['Time']=pd.to_datetime(spi_df['Time'], unit='s')
        spi_df['V_R'] = spi_df['VEL_RTN_SUN'].apply(lambda lst: lst[0])
        spi_df['V_T'] = spi_df['VEL_RTN_SUN'].apply(lambda lst: lst[1])
        spi_df['V_N'] = spi_df['VEL_RTN_SUN'].apply(lambda lst: lst[2])
        spi_df['V'] = np.sqrt(spi_df['V_R']**2 + spi_df['V_T']**2 + spi_df['V_N']**2)
        spi_df['T'] = spi_df['TEMP']*11604.5 # eV  to K
        spi_df['R'] = spi_df['SUN_DIST']/ 149597870.7 # km to AU

        spi_df = spi_df[spi_df['DENS'] < 1000]
        spi_df = spi_df[(spi_df['V'] < 850) & (spi_df['V'] > 200)]

        spi_df.drop(columns='VEL_RTN_SUN', inplace=True)
        #spi_df.drop(columns='DENS', inplace=True)
        spi_df.drop(columns='TEMP', inplace=True)
        spi_df.drop(columns='SUN_DIST', inplace=True)

        spi_df.set_index('Time', inplace=True)
        spi_df = spi_df.resample(rule=cadence).median()

    else:
        print('### NO SPI FILES ###')
        index = pd.to_datetime([timeframe[0]])
        spi_df = pd.DataFrame(index=index,
                            columns=['V_R', 'V_T', 'V_N', 'V', 'T', 'R'])


    #Solar Probe Cup

    print('### EXTRACTING SPC DATA ###')

    if len(spc_files) > 0:

        spc_df = read_cdf_to_df.read_cdf_files_to_dataframe(spc_files, ['Epoch', 'np_moment'])

        spc_df['Time'] = Time(spc_df['Epoch'], format='cdf_tt2000', scale='utc').to_datetime()
        spc_df.set_index('Time', inplace=True)
        spc_df = spc_df[spc_df['np_moment'] > 0.]
        spc_df['N'] = spc_df['np_moment'].copy()
        spc_df.drop(columns='np_moment', inplace = True)
        spc_df.drop(columns='Epoch', inplace = True)

        spc_df = spc_df.resample(rule=cadence).median()

    else:
        print('### NO SPC FILES ###')
        index = pd.to_datetime([timeframe[0]])
        spc_df = pd.DataFrame(index=index, columns=['N'])
        if len(spi_df['DENS'])>1:
            print('### USING SPI DENSITY ###')
            spc_df['N'] = spi_df['DENS']
  
    psp_df = pd.concat([spc_df, spi_df, fields_df], axis=1)


    #Calculate further plasma parameters
    psp_df['P_t'] = (psp_df['N'] * psp_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    psp_df['P_B'] = psp_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    psp_df['P'] = psp_df['P_t'] + psp_df['P_B']
    psp_df['Beta'] = psp_df['P_t'] / psp_df['P_B']
    try:
        psp_df['POL'] = np.sign(psp_df['B_R'] - (psp_df['B_T'] * psp_df['R'] * 2.7 * 10**(-6)) / psp_df['V'])
    except Exception as e:
        psp_df['POL'] = np.sign(-1)
    psp_df['S_P'] = psp_df['T']/psp_df['N']**(2./3.)/11604.5

    psp_df['Spacecraft_ID'] = 1

    return psp_df

def plot(psp_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)

    # Plot multiple time series using seaborn's lineplot in each subplot

    psp_df = psp_df[~psp_df.index.duplicated(keep='first')]


    sns.lineplot(data=psp_df, x=psp_df.index, y='V', ax=axes[0], color='black')
    sns.lineplot(data=psp_df, x=psp_df.index, y='V_R', ax=axes[0], color='blue', alpha=0.5)
    axes[0].set_ylabel('V $[km s^{-1}]$')

    
    sns.lineplot(data=psp_df, x=psp_df.index, y='V_T', ax=axes[1], label='V_T')
    sns.lineplot(data=psp_df, x=psp_df.index, y='V_N', ax=axes[1], label='V_N')
    axes[1].set_ylabel('V_TN $[km s^{-1}]$')

    sns.lineplot(data=psp_df, x=psp_df.index, y='N', ax=axes[2], color='black')
    axes[2].set_ylabel('N $[cm^{-3}]$')


    sns.lineplot(data=psp_df, x=psp_df.index, y='P', ax=axes[3], color='black')
    axes[3].set_ylabel('P $[nPa]$')

    psp_df['polarity'] = ['+' if pol > 0 else '-' for pol in psp_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=psp_df, x=psp_df.index, y='B', ax=axes[4], hue='polarity', palette = colors, s=5, alpha=1)
    axes[4].set_ylabel('B $[nT]$')
    #axes[4].set_ylim([0, 20])

    sns.lineplot(data=psp_df, x=psp_df.index, y='B_R', ax=axes[5], color='red', label='B_R')
    sns.lineplot(data=psp_df, x=psp_df.index, y='B_T', ax=axes[5], color='green', label='B_T')
    sns.lineplot(data=psp_df, x=psp_df.index, y='B_N', ax=axes[5], color='blue', label='B_N')
    axes[5].set_ylabel('B $[nT]$')

    sns.lineplot(data=psp_df, x=psp_df.index, y='S_P', ax=axes[6], color='black')
    axes[6].fill_between(psp_df.index, 2.69, 4, color='grey', alpha=0.7)
    #axes[5].set_ylim([0, 50])
    axes[6].set_ylabel('$S_p$ $[eV cm^{2}]$')

    sns.lineplot(data=psp_df, x=psp_df.index, y='R', ax=axes[7], color='black')
    axes[7].set_ylabel('r $[AU]$')


    ax2 = axes[2].twinx()
    sns.lineplot(data=psp_df, x=psp_df.index, y='T',  ax=ax2, color='tab:blue')
    ax2.set_ylabel('T $[K]$')
    #ax2.set_ylim([0, 2000000])
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')


    ax5 = axes[7].twinx()
    sns.lineplot(data=psp_df, x=psp_df.index, y='LAT', ax=ax5, color='tab:blue')
    #ax5.set_ylim([-10, 50])
    ax5.set_ylabel('LAT $[°]$')
    ax5.spines['right'].set_color('tab:blue')
    ax5.yaxis.label.set_color('tab:blue')
    ax5.tick_params(axis='y', colors='tab:blue')

    ax3 = axes[3].twinx()
    sns.lineplot(data=psp_df, x=psp_df.index, y='Beta', ax=ax3, color='tab:blue')
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


    # #axes[0].set_title('PSP')
    # axes[1].set_title('')
    # axes[2].set_title('')

    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)



def load(month):
        
    from CIRESA import filefinder
    import pandas as pd

    root_dir = 'C:/Users/14milosi/CIRESA/reduced_data/psp'

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
    root_dir = 'C:/Users/14milosi/CIRESA/psp_data/'
    
    dir_fields = root_dir + 'fields/'
    dir_spi = root_dir + 'sweap/spi/'
    dir_spc = root_dir + 'sweap/spc/'

    fields_files = filefinder.find_files_in_timeframe(dir_fields, timeframe[0], timeframe[1])
    spi_files = filefinder.find_files_in_timeframe(dir_spi, timeframe[0], timeframe[1])
    spc_files = filefinder.find_files_in_timeframe(dir_spc, timeframe[0], timeframe[1])

    # Combine all files to delete
    all_files = fields_files + spi_files + spc_files

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


def download_reduce_save_space(month, cadence='0.1H'):

    from CIRESA import psp, filefinder
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if isinstance(month, str):
        month = [month]

    for m in month:

        if pd.to_datetime(m) < pd.to_datetime('2018-10') :
            print('### NO PSP DATA BEFORE 2018-10 ###')
        else:

            print (m)

            if os.path.exists('reduced_data\psp\psp_data'+m+'.parquet'):
                psp_df = psp.load(m)

            else:
                timeframe = filefinder.get_month_dates(m)

                psp.download(timeframe)
                psp_df = psp.reduce(timeframe, cadence)
                psp_df.to_parquet('reduced_data\psp\psp_data'+m+'.parquet')

        try:
            # Plot and save the figure
            psp.plot(psp_df)
            plt.savefig(f'psp_data/monthly_plots/psp_{m}.png')
            plt.close()  # Close the plot to free up memory
        except KeyError as e:
            print(f"Error plotting data for {m}: {e}")
        finally:
            # Ensure psp.delete() is called regardless of success or failure
            psp.delete(m)

