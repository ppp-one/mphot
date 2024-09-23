import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def interpolate_dfs(index, *data):
    '''
    Interpolates panda dataframes onto an index, of same index type (e.g. wavelength in microns)

    Parameters
    ----------
    index: 1d array which data is to be interpolated onto
    data:       Pandas dataframes 

    Returns
    -------
    df: Interpolated dataframe

    '''

    df = pd.DataFrame({'tmp': index}, index=index)
    for dat in data:
        dat = dat[~dat.index.duplicated(keep='first')]
        df = pd.concat([df, dat], axis=1)
    df = df.sort_index()
    df = df.interpolate('index').reindex(index)
    df = df.drop('tmp', 1)

    return df


wavelengths = np.arange(0.5, 2, 0.0001)

mirror = pd.read_csv('datafiles/systems/optics/Al_mirror.csv', index_col=0, header=None, names=['mirror'])
lens = pd.read_csv('datafiles/systems/optics/fused_silica_lens.csv', index_col=0, header=None, names=['lens'])
qe = pd.read_csv('datafiles/systems/qe/PIRTqe_-60.csv', index_col=0, header=None, names=['qe'])
filter = pd.read_csv('datafiles/filters/zYJ.csv', index_col=0, header=None, names=['filter'])

inp = interpolate_dfs(wavelengths, mirror, lens, qe, filter)

inp['total'] = inp['mirror']**2 * inp['lens']**2 * inp['qe'] * inp['filter']

inp.plot()


# compare to current SR
current = pd.read_csv('datafiles/SRs/pirtSPC_-60_zYJ_instrumentSR.csv', index_col=0, header=None, names=['sr'])

plt.figure()
plt.plot(inp.index, inp['total'], label='total')
plt.plot(current.index, current['sr'], label='current')
plt.legend()

plt.show()