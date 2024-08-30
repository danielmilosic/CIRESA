def download_stereo_a(timeframe):
    import pyspedas

    pyspedas.stereo.mag(trange=timeframe, time_clip=True, get_support_data=True
                    , downloadonly=True)
    pyspedas.stereo.plastic(trange=timeframe, time_clip=True, get_support_data=True, level = 'l2', datatype='10min'
                    , downloadonly=True)