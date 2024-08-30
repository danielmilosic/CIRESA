def download_solo(timeframe):
    import pyspedas

    pyspedas.solo.swa(trange=timeframe, time_clip=True, level='l2', datatype='pas-grnd-mom', get_support_data=True
                 , downloadonly=True)
    pyspedas.solo.swa(trange=timeframe, time_clip=False, level='l3', datatype='his-comp-10min', get_support_data=True
                  , downloadonly=True)
    pyspedas.solo.mag(trange=timeframe, time_clip=True, get_support_data=True
                 , downloadonly=True)