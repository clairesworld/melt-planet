import pyMelt as m
import numpy as np
import matplotlib.pyplot as plt
from MLTMantle import years2sec, get_Mantle_struct
import pandas as pd
import pickle as pkl

import sys
sys.path
sys.path.append('/home/claire/Works/rocky-water/py/')
from useful_and_bespoke import colorize

def test_solidii(name, outputpath=None, p_max_melt=10, p_max_plot=22):
    from test_results import plot_pickle_timesteps
    from MLTMantle import get_Mantle_struct

    man = get_Mantle_struct()
    P_GPa = man.P * 1e-9

    fig, ax = plt.subplots(1, 1)

    # plot temperature solution
    import pickle as pkl
    import glob

    # tmp pickle files
    files = glob.glob('output/tests/tmp/' + name + '.pkl*')
    data, t = [], []
    for ii, f in enumerate(files):  # all timesteps
        with open(f, "rb") as pfile:
            try:
                d = pkl.load(pfile)
                data.append(d['u'])
                t.append(d['t'] / years2sec * 1e-9)
            except EOFError:
                pass

    # need to sort before plotting colour code
    t, data = zip(*sorted(zip(t, data)))
    # alphas = np.linspace(0, 0.5, len(t))
    c = colorize(t, cmap='Greys')[0]
    for ii, tt in enumerate(t):
        ax.plot(data[ii], P_GPa, c=c[ii], alpha=0.5)

    # init lithologies
    lz = m.lithologies.matthews.klb1()
    px = m.lithologies.matthews.kg1()
    hz = m.lithologies.shorttle.harzburgite()

    hlz = m.hydrousLithology(lz, 0.1, continuous=True,
                             phi=0.5)  # continuous melting - only matters for hydrous melting because water is extracted
    hlz_batch = m.hydrousLithology(lz, 0.1)

    # put into pymelt mantle object - mostly lherzolite mantle
    mantle = m.mantle([lz, px, hz], [6, 2, 2], ['Lz', 'Px', 'Hz'])

    # plot pymelt solidii
    p = np.linspace(0, p_max_melt, 100)
    t = np.zeros([len(p), 4])
    for i in range(len(p)):
        t[i, 0] = lz.TSolidus(p[i]) + 273
        t[i, 1] = hlz.TSolidus(p[i]) + 273
        t[i, 2] = hlz_batch.TSolidus(p[i]) + 273
        t[i, 3] = lz.TLiquidus(p[i]) + 273

    ax.plot(t[:, 0], p, label='Anhydrous')
    ax.plot(t[:, 1], p, label='0.1 wt%')
    ax.plot(t[:, 2], p, label='0.1 wt% batch')
    ax.plot(t[:, 3], p, label='Liquidus')

    ax.set_xlabel('T (K)')
    ax.set_ylabel('P (GPa)')
    ax.invert_yaxis()
    ax.set_ylim(22, 0)
    ax.legend()
    plt.show()


test_solidii(name='Tachinami_buffered2')



