def download_psp(timeframe):
    import pyspedas

    pyspedas.psp.fields(trange=timeframe, time_clip=True, datatype='mag_RTN', get_support_data=True
                    , downloadonly=True, level='l2')
    pyspedas.psp.spi(trange=timeframe, time_clip=True, get_support_data=True
                 , downloadonly=True
                 , datatype='sf00_l3_mom'
                 )
    pyspedas.psp.spc(trange=timeframe, downloadonly=True, datatype='l3i')

def reduce_psp(timeframe, cadence):

    from CIRESA import filefinder, read_cdf_to_df
    import pandas as pd
    import numpy as np
    from astropy.time import Time

    root_dir = 'C:/Users/14milosi/CIRESA/psp_data/'
    
    dir_fields = root_dir + 'fields/'
    dir_spi = root_dir + 'sweap/spi/'
    dir_spc = root_dir + 'sweap/spc/'

    fields_files = filefinder.find_files_in_timeframe(dir_fields, '2022-10-10', '2022-10-11')
    spi_files = filefinder.find_files_in_timeframe(dir_spi, '2022-10-10', '2022-10-11')
    spc_files = filefinder.find_files_in_timeframe(dir_spc, '2022-10-10', '2022-10-11')

    print(fields_files, spc_files, spi_files)



    fields_df = read_cdf_to_df.read_cdf_files_to_dataframe(fields_files, ['epoch_mag_RTN', 'psp_fld_l2_mag_RTN'])

    # Resample and drop unnecessary columns
    fields_df = fields_df.iloc[::10]
    fields_df['Time'] = Time(fields_df['epoch_mag_RTN'], format='cdf_tt2000', scale='utc').to_datetime()
    fields_df.set_index('Time', inplace=True)
    fields_df['B_R'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[0])
    fields_df['B_T'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[1])
    fields_df['B_N'] = fields_df['psp_fld_l2_mag_RTN'].apply(lambda lst: lst[2])
    fields_df.drop('psp_fld_l2_mag_RTN', axis=1, inplace=True)
    fields_df.drop('epoch_mag_RTN', axis=1, inplace=True)

    fields_df = fields_df.resample(rule=cadence).median(cadence)

    fields_df['B'] = np.sqrt(fields_df['B_R']**2 + fields_df['B_T']**2 + fields_df['B_N']**2)


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
    spi_df.drop(columns='DENS', inplace=True)
    spi_df.drop(columns='TEMP', inplace=True)
    spi_df.drop(columns='SUN_DIST', inplace=True)

    spi_df.set_index('Time', inplace=True)
    spi_df = spi_df.resample(rule=cadence).median(cadence)


    spc_df = read_cdf_to_df.read_cdf_files_to_dataframe(spc_files, ['Epoch', 'np_moment'])

    spc_df['Time'] = Time(spc_df['Epoch'], format='cdf_tt2000', scale='utc').to_datetime()
    spc_df.set_index('Time', inplace=True)
    spc_df = spc_df[spc_df['np_moment'] > 0.]
    spc_df['N'] = spc_df['np_moment'].copy()
    spc_df.drop(columns='np_moment', inplace = True)
    spc_df.drop(columns='Epoch', inplace = True)
                
    spc_df = spc_df.resample(rule=cadence).median(cadence)
  
    psp_df = pd.concat([spc_df, spi_df, fields_df], axis=1)

    psp_df['P_t'] = (psp_df['N'] * psp_df['V']**2) / 10**19 / 1.6727   * 10**6 *10**9 # J/cm^3 to nPa
    psp_df['P_B'] = psp_df['B']**2 / 2. / 1.25663706212*10**(-6) / 10**9    * 10**6 *10**9 #nT to T # J/cm^3 to nPa
    psp_df['P'] = psp_df['P_t'] + psp_df['P_B']
    psp_df['Beta'] = psp_df['P_t'] / psp_df['P_B']
    psp_df['POL'] = np.sign(psp_df['B_R'] - psp_df['B_T']*psp_df['R']*2.7*10**(-6)/psp_df['V'])
    psp_df['S_P'] = psp_df['T']/psp_df['N']**(2./3.)/11604.5

    return psp_df

def plot_psp(psp_df):
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots with specified layout
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 12), sharex=True)

    # Plot multiple time series using seaborn's lineplot in each subplot


    sns.lineplot(data=psp_df, x=psp_df.index, y='V', ax=axes[0], color='black')
    axes[0].set_ylabel('V $[km s^{-1}]$')

    sns.lineplot(data=psp_df, x=psp_df.index, y='V_R', ax=axes[1], label='V_R')
    axes[1].set_ylabel('V_RTN $[km s^{-1}]$')
    sns.lineplot(data=psp_df, x=psp_df.index, y='V_T', ax=axes[1], label='V_T')
    sns.lineplot(data=psp_df, x=psp_df.index, y='V_N', ax=axes[1], label='V_N')


    sns.lineplot(data=psp_df, x=psp_df.index, y='N', ax=axes[2], color='black')
    axes[2].set_ylabel('N $[cm^{-3}]$')


    sns.lineplot(data=psp_df, x=psp_df.index, y='P', ax=axes[3], color='black')
    axes[3].set_ylabel('P $[nPa]$')

    psp_df['polarity'] = ['+' if pol > 0 else '-' for pol in psp_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=psp_df, x=psp_df.index, y='B', ax=axes[4], hue='polarity', palette = colors, s=5, alpha=1)
    axes[4].set_ylabel('B $[nT]$')
    axes[4].set_ylim([0, 20])

    sns.lineplot(data=psp_df, x=psp_df.index, y='S_P', ax=axes[5], color='black')
    axes[5].fill_between(psp_df.index, 2.69, 4, color='grey', alpha=0.7)
    #axes[5].set_ylim([0, 50])
    axes[5].set_ylabel('$S_p$ $[eV cm^{2}]$')

# sns.lineplot(data=psp, x=psp_df.index, y='SUN_DIST', ax=axes[5], color='black')
# axes[5].set_ylabel('r $[AU]$')
# axes[5].set_ylim([0.755,0.762])


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


# linewidth = 1
# for i in range(6):
#     axes[i].fill_between(['2022-10-21 17:30', '2022-10-21 18:30'], 0, 700, color='blue', alpha=0.5)
#     #axes[i].fill_between(['2022-10-21 06:00', '2022-10-21 17:00'], 0, 700, color='orange', alpha=0.5)
#     axes[i].fill_between(['2022-10-21 18:30', '2022-10-22 05:00'], 0, 700, color='orange', alpha=0.5)
#     axes[i].fill_between(['2022-10-22 05:00', '2022-10-22 14:30'], 0, 700, color='red', alpha=0.5)

#     axes[i].fill_between(['2022-10-22 14:30', '2022-10-23 09:00'], 0, 700, color='grey', alpha=0.5)

#     axes[i].fill_between(['2022-10-23 09:00', '2022-10-23 22:30'], 0, 700, color='orange', alpha=0.5)
#     axes[i].fill_between(['2022-10-23 22:30', '2022-10-24 12:00'], 0, 700, color='red', alpha=0.5)
#     axes[i].fill_between(['2022-10-24 12:00', '2022-10-26 22:00'], 0, 700, color='grey', alpha=0.5)
#     #axes[i].fill_between(['2022-10-26 01:00', '2022-10-26 06:30'], 0, 700, color='grey', alpha=0.5)
#     # axes[i].axvline(x=pd.to_datetime('2022-10-23 18:30'), color='black', linestyle='-', linewidth=linewidth)
#     # axes[i].axvline(x=pd.to_datetime('2022-10-23 22:30'), color='black', linestyle='-', linewidth=linewidth)
#     # axes[i].axvline(x=pd.to_datetime('2022-10-24 01:00'), color='black', linestyle='-', linewidth=linewidth)

# sns.scatterplot(data=psp, x=psp_df.index, y='B', ax=axes[3], hue='polarity', palette = colors, s=10, alpha=1, legend=False, linewidth=0.4)
# axes[1].set_xlim([pd.to_datetime('2022-10-20'), pd.to_datetime('2022-10-28')])

    ax2 = axes[2].twinx()
    sns.lineplot(data=psp_df, x=psp_df.index, y='T',  ax=ax2, color='tab:blue')
    ax2.set_ylabel('T $[K]$')
    #ax2.set_ylim([0, 2000000])
    ax2.spines['right'].set_color('tab:blue')
    ax2.yaxis.label.set_color('tab:blue')
    ax2.tick_params(axis='y', colors='tab:blue')


# ax5 = axes[5].twinx()
# sns.lineplot(data=psp, x=psp_df.index, y='LAT', ax=ax5, color='tab:blue')
# #ax5.set_ylim([-10, 50])
# ax5.set_ylabel('LAT $[°]$')
# ax5.spines['right'].set_color('tab:blue')
# ax5.yaxis.label.set_color('tab:blue')
# ax5.tick_params(axis='y', colors='tab:blue')

# ax3 = axes[2].twinx()
# sns.lineplot(data=psp, x=psp_df.index, y='Beta', ax=ax3, color='tab:blue')
# #ax5.set_ylim([-10, 50])
# ax3.set_ylabel(r'$\beta$')
# ax3.set_yscale('log')
# ax3.spines['right'].set_color('tab:blue')
# ax3.yaxis.label.set_color('tab:blue')
# ax3.tick_params(axis='y', colors='tab:blue')


# fig.lines.append(plt.Line2D([0.481, 0.509], [.995, 0.995], transform=fig.transFigure, color="red", linewidth=1))
# fig.lines.append(plt.Line2D([0.481, 0.481], [.99, 1], transform=fig.transFigure, color="red", linewidth=1))
# fig.lines.append(plt.Line2D([0.509, 0.509], [.99, 1], transform=fig.transFigure, color="red", linewidth=1))
# fig.lines.append(plt.Line2D([0.4962, 0.4962], [.99, 1], transform=fig.transFigure, color="red", linewidth=1))

# fig.lines.append(plt.Line2D([0.801, 0.738], [.995, 0.995], transform=fig.transFigure, color="black", linewidth=1))
# fig.lines.append(plt.Line2D([0.801, 0.801], [.99, 1], transform=fig.transFigure, color="black", linewidth=1))
# fig.lines.append(plt.Line2D([0.738, 0.738], [.99, 1], transform=fig.transFigure, color="black", linewidth=1))


# plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1)
