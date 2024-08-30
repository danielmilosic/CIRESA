def download_omni(timeframe):
    import pyspedas

    pyspedas.omni.data(trange=timeframe, time_clip=True, datatype='1min', get_support_data=True
                , downloadonly=True)
