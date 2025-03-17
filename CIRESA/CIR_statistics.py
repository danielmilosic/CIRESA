import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import CIRESA
import CIRESA.CIR_statistics 

def positions(CIR):
    # Calculate the positions of each interface of an identified CIR


    CIR = CIR[['R', 'CARR_LON', 'LAT', 'Interface', 'Interface_uncertainty']]
    positions = []

    if len(CIR[CIR['Interface']==1])>0:
        Forward_wave = CIR[CIR['Interface']==1]
        Forward_wave_uncertainty = CIR[CIR['Interface_uncertainty']==1.5]
        positions.append(Forward_wave)
        positions.append(Forward_wave_uncertainty)
    if len(CIR[CIR['Interface']==2])>0:
        Stream_Interface = CIR[CIR['Interface']==2]
        Stream_Interface_uncertainty = CIR[CIR['Interface_uncertainty']==2.5]
        positions.append(Stream_Interface)
        positions.append(Stream_Interface_uncertainty)
    if len(CIR[CIR['Interface']==3])>0:
        Back_wave = CIR[CIR['Interface']==3]
        Back_wave_uncertainty = CIR[CIR['Interface_uncertainty']==3.5]
        positions.append(Back_wave)
        positions.append(Back_wave_uncertainty)
    if len(CIR[CIR['Interface']==4])>0:
        Trailing_edge = CIR[CIR['Interface']==4]
        Trailing_edge_uncertainty = CIR[CIR['Interface_uncertainty']==4.5]
        positions.append(Trailing_edge)
        positions.append(Trailing_edge_uncertainty)

    positions = pd.concat(positions)

    return positions

def find_solar_source(df):

    # Calculate the solar source position in DEG

    solar_rot = 0.0096 /np.pi * 180  #rad/hour to deg/hour
    au = 149597870 #km
    travel_time = (df['R'] - 0.00465)*au / df['V'] /3600 # hours

    solar_source = df['CARR_LON'] - travel_time*solar_rot

    return solar_source

# def plasma_parameters(CIR):
#     # Calculate plasma parameter max/min/mean
#     # for each region of an identified CIR

# def regions_extent(CIR):
#     # Calculate the extent
#     # for each region of an identified CIR


def plot_statistics(CIR, superposed=True, only_si=False):
     # Plot plasma parameters max/min/mean
     # for each region of an identified CIR
    
    if superposed:
        CIR['DATE'] = CIR.index
        CIR.reset_index(inplace=True)
        solar_source = CIRESA.CIR_statistics.find_solar_source(CIR)
        CIR['CARR_LON'] = CIR['CARR_LON'] - solar_source
        CIR.set_index('DATE', inplace=True)

    positions = CIRESA.CIR_statistics.positions(CIR)

    si = positions[positions['Interface']==2]
    sns.scatterplot(data=si, x='CARR_LON', y='R', color = 'k')
    if not only_si:
        fw = positions[positions['Interface']==1]
        sns.scatterplot(data=fw, x='CARR_LON', y='R', color = 'r')
        bw = positions[positions['Interface']==3]
        sns.scatterplot(data=bw, x='CARR_LON', y='R', color = 'r')
        te = positions[positions['Interface']==4]
        sns.scatterplot(data=te, x='CARR_LON', y='R', color = 'orange')

    plt.xlim(0,180)
    plt.ylim(0,2)

    plt.tight_layout(pad=1., w_pad=1., h_pad=.1)
    plt.show()