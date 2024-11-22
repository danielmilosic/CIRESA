import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.dates import DateFormatter
import keyword

def suggest(spacecraft_df, si=None, fw=None, bw=None, te=None):
    ### Prepare suggestions
    spacecraft_df = spacecraft_df[['V', 'N', 'P', 'CARR_LON', 'S_P']].sort_index()

    mean = spacecraft_df.mean()

    # fill NaN values with the mean of each column
    spacecraft_df.fillna(mean, inplace=True)

    # Find the indices based on max, min values
    max_v_index = spacecraft_df['V'].idxmax()
    max_n_index = spacecraft_df['N'].idxmax()
    low_p = 0.5*(np.max(spacecraft_df['P'])-np.min(spacecraft_df['P']))

    # Stream interface
    stream_interface = {
        'time': max_n_index,
        'carr_lon': spacecraft_df.loc[max_n_index, 'CARR_LON']
    }
    
    if si is not None:
        stream_interface = {
        'time': si,
        'carr_lon': spacecraft_df.loc[si, 'CARR_LON']
    }
    
    # Trailing edge
    HSS = spacecraft_df[max_v_index:]
    notHSS = HSS[HSS['S_P']<4.]
    #print('notHSS:', notHSS)
    #trailing_edge_index = notHSS.index[0]

    if len(notHSS)<2:
        trailing_edge_index = spacecraft_df.index[-1]
    else: 
        trailing_edge_index = notHSS.index[0]

    if trailing_edge_index < max_n_index:
        trailing_edge_index = spacecraft_df.index[-1]

    trailing_edge = {
        'time': trailing_edge_index,
        'carr_lon': spacecraft_df.loc[trailing_edge_index, 'CARR_LON']
    }

    if te is not None:
        trailing_edge = {
        'time': te,
        'carr_lon': spacecraft_df.loc[te, 'CARR_LON']
    }
    

    #print('TE:', trailing_edge)
    #print('SI:', stream_interface)

    # Back wave   

    perturbed_fast_wind = spacecraft_df[stream_interface['time']:trailing_edge['time']]
    back_wave_index = perturbed_fast_wind.index[-1] 
    perturbed_fast_wind = perturbed_fast_wind[perturbed_fast_wind['P'] > low_p]
    
    if len(perturbed_fast_wind)>2:
        back_wave_index = perturbed_fast_wind.index[-1]

    back_wave = {
        'time': back_wave_index,
        'carr_lon': spacecraft_df.loc[back_wave_index, 'CARR_LON']
    }
    
    if bw is not None:
        back_wave = {
        'time': bw,
        'carr_lon': spacecraft_df.loc[bw, 'CARR_LON']
    }
        
    #print('BW:', back_wave)
    # Front wave

    perturbed_slow_wind = spacecraft_df[:stream_interface['time']] 
    front_wave_index = perturbed_slow_wind.index[0]
    perturbed_slow_wind = perturbed_slow_wind[perturbed_slow_wind['P']> low_p]

    if len(perturbed_slow_wind)>2:
        front_wave_index = perturbed_slow_wind.index[0]

    front_wave = {
        'time': front_wave_index,
        'carr_lon': spacecraft_df.loc[front_wave_index, 'CARR_LON']
    }
    
    if fw is not None:
        front_wave = {
        'time': fw,
        'carr_lon': spacecraft_df.loc[fw, 'CARR_LON']
    }

    #print('FW:', front_wave)

    return stream_interface, front_wave, back_wave, trailing_edge


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RectangleSelector, SpanSelector

def plot_and_choose_CIR(spacecraft_df):
    matplotlib.use('TkAgg')

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 10), sharex=True)
    
    # Initial plots
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='V', ax=axes[0], color='black')
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='N', ax=axes[1], color='black')
    axes[1].set_ylabel('N $[cm^{-3}]$')
    
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='P', ax=axes[2], color='black')
    axes[2].set_ylabel('P $[nPa]$')
    
    spacecraft_df['polarity'] = ['+' if pol > 0 else '-' for pol in spacecraft_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=spacecraft_df, x=spacecraft_df.index, y='B', ax=axes[3], hue='polarity', palette=colors, s=5, alpha=1)
    axes[3].set_ylabel('B $[nT]$')
    
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_R', ax=axes[4], color='red', label='B_R')
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_T', ax=axes[4], color='green', label='B_T')
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_N', ax=axes[4], color='blue', label='B_N')
    axes[4].set_ylabel('B $[nT]$')
    
    sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='S_P', ax=axes[5], color='black')
    axes[5].fill_between(spacecraft_df.index, 2.69, 4, color='grey', alpha=0.7)
    axes[5].set_ylabel('$S_p$ $[eV cm^{2}]$')
    plt.title('Please choose a time interval')

    df_sorted = spacecraft_df.sort_index()
    axes[1].set_xlim(df_sorted.index[0],df_sorted.index[-1] )
    plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1) 
    filtered_df_container = []
    def onselect(x_min, x_max):
        """ Callback function for span selection """
        # Redraw the figure with the selected time range
        plt.close()
        x1, x2 = axes[0].get_xlim()
        lower_lim = (x_min - x1) / (x2 - x1) * len(spacecraft_df)
        upper_lim = (x_max - x1) / (x2 - x1) * len(spacecraft_df)

        filtered_df = spacecraft_df.iloc[int(lower_lim.round()):int(upper_lim.round())]

        # Store the filtered DataFrame
        si, fw, bw, te = suggest(filtered_df)

        print(si, fw, bw, te)

        filtered_df_container.clear()
        filtered_df_container.append(spacecraft_df[fw['time'] - pd.Timedelta('1D') : te['time'] + pd.Timedelta('1D')])


    # Create the SpanSelector
    span_selector = SpanSelector(axes[0], onselect, 'horizontal', useblit=True,
                                  minspan=0, interactive=True)

    plt.show()

    CIR = plot_and_identify_CIR_Interfaces(filtered_df_container[0])
    return CIR
    #return filtered_df_container[0] if filtered_df_container else None


def plot_and_identify_CIR_Interfaces(spacecraft_df):
    
    matplotlib.use('TkAgg')
    spacecraft_df['Interface'] = spacecraft_df['V']*0
    spacecraft_df['Interface_uncertainty'] = spacecraft_df['V']*0


    def plot_Interfaces(spacecraft_df, title, si=None, fw=None, bw=None, te=None, span = True, Interface = 0):
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 10), sharex=True)
        
        si, fw, bw, te = suggest(spacecraft_df, si, fw, bw, te)

        # Initial plots
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='V', ax=axes[0], color='black')
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='N', ax=axes[1], color='black')
        axes[1].set_ylabel('N $[cm^{-3}]$')
        
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='P', ax=axes[2], color='black')
        axes[2].set_ylabel('P $[nPa]$')
        
        spacecraft_df['polarity'] = ['+' if pol > 0 else '-' for pol in spacecraft_df['POL']]
        colors = {'-': 'tab:blue', '+': 'tab:red'}
        sns.scatterplot(data=spacecraft_df, x=spacecraft_df.index, y='B', ax=axes[3], hue='polarity', palette=colors, s=5, alpha=1)
        axes[3].set_ylabel('B $[nT]$')
        
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_R', ax=axes[4], color='red', label='B_R')
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_T', ax=axes[4], color='green', label='B_T')
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='B_N', ax=axes[4], color='blue', label='B_N')
        axes[4].set_ylabel('B $[nT]$')
        
        sns.lineplot(data=spacecraft_df, x=spacecraft_df.index, y='S_P', ax=axes[5], color='black')
        axes[5].fill_between(spacecraft_df.index, 2.69, 4, color='grey', alpha=0.7)
        axes[5].set_ylabel('$S_p$ $[eV cm^{2}]$')
        axes[0].set_title(title)

        for ax in axes:
                ax.axvline(x=fw['time'], color='red')
                ax.axvline(x=si['time'], color='black')
                ax.axvline(x=bw['time'], color='red')
                ax.axvline(x=te['time'], color='orange')
                
        formatter = DateFormatter("%y-%m-%d %H:%M")
        axes[-1].xaxis.set_major_formatter(formatter)
        df_sorted = spacecraft_df.sort_index()
        axes[1].set_xlim(df_sorted.index[0],df_sorted.index[-1] )
        plt.tight_layout(pad=1., w_pad=0.5, h_pad=.1) 


        if span:
            def onselect(x_min, x_max):
                """ Callback function for rectangle selection """
                # Redraw the figure with the selected time range
                ax = plt.gca()  # Get the current axes
                x1, x2 = ax.get_xlim()
                lower_lim = (x_min-x1)/(x2-x1)*len(spacecraft_df)
                upper_lim = (x_max-x1)/(x2-x1)*len(spacecraft_df)

                spacecraft_df['Interface_uncertainty'].iloc[[int(lower_lim.round()),int(upper_lim.round())]] = Interface


                plt.close()


            # Create the SpanSelector
            span_selector = SpanSelector(axes[0], onselect, 'horizontal', useblit=True,
                                        minspan=0, interactive=True)
            
        else:
            def onclick(event):
                """ Callback function to select a single x value """
                if event.inaxes == axes[0]:  # Only handle clicks in the first plot

                    x_value = event.xdata

                    # Get the index of the closest value
                    x1, x2 = axes[0].get_xlim()
                    x = (x_value - x1) / (x2 - x1) * len(spacecraft_df)
                    index = int(x.round())
                    

                    spacecraft_df['Interface'].iloc[index:index + 1] = Interface

                    plt.close()
            
            # Connect the click event to the `onclick` function
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            

        plt.show()
    
    plot_Interfaces(spacecraft_df, 'First, choose the Stream Interface'
                 , span = False, Interface=2)
    plot_Interfaces(spacecraft_df, 'Give an uncertainty range for the Stream Interface'
                 , Interface=2.5, si = spacecraft_df[spacecraft_df['Interface']==2].index[0])
    plot_Interfaces(spacecraft_df, 'Second, choose the Forward Wave'
                 , span = False, Interface=1
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0])
    plot_Interfaces(spacecraft_df, 'Give an uncertainty range for the Forward Wave'
                 , Interface=1.5
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
                 , fw = spacecraft_df[spacecraft_df['Interface']==1].index[0])
    plot_Interfaces(spacecraft_df, 'Third, choose the Back Wave'
                 , span = False, Interface=3
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
                 , fw = spacecraft_df[spacecraft_df['Interface']==1].index[0])
    plot_Interfaces(spacecraft_df, 'Give an uncertainty range for the Back Wave'
                 , Interface=3.5
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
                 , fw = spacecraft_df[spacecraft_df['Interface']==1].index[0]
                 , bw = spacecraft_df[spacecraft_df['Interface']==3].index[0]
                 )
    plot_Interfaces(spacecraft_df, 'Last, choose the Trailing edge'
                 , span = False, Interface=4
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
                 , fw = spacecraft_df[spacecraft_df['Interface']==1].index[0]
                 , bw = spacecraft_df[spacecraft_df['Interface']==3].index[0]
                 )
    plot_Interfaces(spacecraft_df, 'Give an uncertainty range for the Trailing edge'
                 , Interface=4.5
                 , si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
                 , fw = spacecraft_df[spacecraft_df['Interface']==1].index[0]
                 , bw = spacecraft_df[spacecraft_df['Interface']==3].index[0]
                 , te = spacecraft_df[spacecraft_df['Interface']==4].index[0])
    
    fw = spacecraft_df[spacecraft_df['Interface']==1].index[0]
    si = spacecraft_df[spacecraft_df['Interface']==2].index[0]
    bw = spacecraft_df[spacecraft_df['Interface']==3].index[0]
    te = spacecraft_df[spacecraft_df['Interface']==4].index[0]

    CIR = spacecraft_df[fw - pd.Timedelta('1D') : te + pd.Timedelta('1D')]
    CIR['Region'] = CIR['Interface']*0
    CIR.loc[fw:si, 'Region'] = 1
    CIR.loc[si:bw, 'Region'] = 2
    CIR.loc[bw:te, 'Region'] = 3

    plot_identified_CIR(CIR)
                 
    return CIR

def plot_identified_CIR(filtered_df):

    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 10), sharex=True)
    
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='V', ax=axes[0], color='black')
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='N', ax=axes[1], color='black')
    axes[1].set_ylabel('N $[cm^{-3}]$')
    
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='P', ax=axes[2], color='black')
    axes[2].set_ylabel('P $[nPa]$')

    filtered_df['polarity'] = ['+' if pol > 0 else '-' for pol in filtered_df['POL']]
    colors = {'-': 'tab:blue', '+': 'tab:red'}
    sns.scatterplot(data=filtered_df, x=filtered_df.index, y='B', ax=axes[3], hue='polarity', palette=colors, s=5, alpha=1)
    axes[3].set_ylabel('B $[nT]$')
    
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='B_R', ax=axes[4], color='red', label='B_R')
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='B_T', ax=axes[4], color='green', label='B_T')
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='B_N', ax=axes[4], color='blue', label='B_N')
    axes[4].set_ylabel('B $[nT]$')
    
    sns.lineplot(data=filtered_df, x=filtered_df.index, y='S_P', ax=axes[5], color='black')
    axes[5].fill_between(filtered_df.index, 2.69, 4, color='grey', alpha=0.7)
    axes[5].set_ylabel('$S_p$ $[eV cm^{2}]$')

    formatter = DateFormatter("%y-%m-%d %H:%M")
    df_sorted = filtered_df.sort_index()
    axes[1].set_xlim(df_sorted.index[0],df_sorted.index[-1] )

    fw = filtered_df[filtered_df['Interface']==1].index[0]
    si = filtered_df[filtered_df['Interface']==2].index[0]
    bw = filtered_df[filtered_df['Interface']==3].index[0]
    te = filtered_df[filtered_df['Interface']==4].index[0]

    for ax in axes:
        ax.axvline(x=fw, color='red')
        ax.axvline(x=si, color='black')
        ax.axvline(x=bw, color='red')
        ax.axvline(x=te, color='orange')

        ymin, ymax = ax.get_ylim()
  
        ax.fill_between([fw, si], ymin, ymax, color='orange', alpha=0.5)
        ax.fill_between([si, bw], ymin, ymax, color='red', alpha=0.5)
        ax.fill_between([bw, te], ymin, ymax, color='grey', alpha=0.5)

    plt.show()

from CIRESA.utils import spacecraft_ID
def save_to_CIR_data_base(CIR):

    spacecraft = spacecraft_ID(CIR)
    day1 = CIR.index[0].strftime('%Y-%m-%d')
    lastday = CIR.index[-1].strftime('%Y-%m-%d')
    name = spacecraft+day1 + '---' + lastday

    CIR.to_parquet('CIR_Database/'
                   + name + 
                   '.parquet')

def load_CIR(month, concat=False):
    from CIRESA import filefinder

    root_dir = 'CIR_Database'

    files = filefinder.find_parquet_files(root_dir, month)
    # Ensure 'files' is always a list, even if a single file path is returned
    if isinstance(files, str):
        files = [files]


    CIR = []
    for f in files:

        print(f)
        df = pd.read_parquet(f)
        CIR.append(df)

    if concat:
        return pd.concat(CIR)
    else:
        return CIR
