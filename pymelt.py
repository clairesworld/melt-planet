import pyMelt as m
import numpy as np
import matplotlib.pyplot as plt
from MLTMantle import years2sec, get_Mantle_struct
import pandas as pd
import pickle as pkl

import sys
sys.path
sys.path.append('/home/claire/Works/rocky-water/py/')
from useful_and_bespoke import colorize, colourbar


def potential_temperature(T, P, alpha, rho, cp):
    return T / np.exp(alpha/(rho * cp) * P)


def plot_temperature_evolution(name, P_GPa, outputpath=None, fig=None, ax=None, final=False):
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
                t.append(d['t'] / years2sec * 1e-9)  # Gyr
            except EOFError:
                pass

    # need to sort before plotting colour code
    t, data = zip(*sorted(zip(t, data)))

    if not final:
        # alphas = np.linspace(0, 0.5, len(t))
        c = colorize(t, cmap='Greys')[0]
        for ii in np.arange(len(t))[::100]:
            ax.plot(data[ii], P_GPa, c=c[ii], alpha=0.9)
        cbar = colourbar(vector=np.array(t), ax=ax, label='t (Gyr)', cmap='Greys')
    else:
        ax.plot(data[-1], P_GPa, c='k', alpha=0.9)
    return fig, ax, data[-1]

def test_solidii(name, outputpath=None, p_max_melt=10, p_max_plot=22):
    from MLTMantle import get_Mantle_struct

    man = get_Mantle_struct()
    P_GPa = man.P * 1e-9

    fig, ax = plt.subplots(1, 1)

    # plot temperature solution
    fig, ax, _ = plot_temperature_evolution(name, P_GPa, outputpath=outputpath, fig=fig, ax=ax)

    # init lithologies
    lz = m.lithologies.matthews.klb1()
    px = m.lithologies.matthews.kg1()
    hz = m.lithologies.shorttle.harzburgite()

    hlz = m.hydrousLithology(lz, 0.1, continuous=True,
                             phi=0.5)  # continuous melting - only matters for hydrous melting because water is extracted
    hlz_batch = m.hydrousLithology(lz, 0.1)

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


def decompression_melting(name, outputpath=None, p_max_melt=10, p_max_plot=22):
    from MLTMantle import get_Mantle_struct

    man = get_Mantle_struct(Nm=10000)
    P_GPa = man.P * 1e-9

    fig, ax = plt.subplots(1, 1)

    # plot temperature solution for final
    fig, ax, T_soln = plot_temperature_evolution(name, P_GPa, outputpath=outputpath, fig=fig, ax=ax, final=True)

    # get potential temperature using T, P at 10 GPa
    i_10 = (np.abs(P_GPa - 10)).argmin()

    Tp = potential_temperature(T_soln[i_10], P_GPa[i_10] * 1e9, man.alpha_m[i_10], man.rho_m[i_10], man.cp_m[i_10])
    Tp_C = Tp - 273.15
    print('Tp = ', Tp, 'K', 'at 10 GPa', T_soln[i_10], 'K')

    # print('properties',man.rho_m[i_10], man.alpha_m[i_10] * 1e6, man.cp_m[i_10])

    # init lithologies
    lz = m.lithologies.matthews.klb1(rhos=man.rho_m[i_10] * 1e-3, alphas=man.alpha_m[i_10] * 1e6, CP=man.cp_m[i_10])
    px = m.lithologies.matthews.kg1(rhos=man.rho_m[i_10] * 1e-3, alphas=man.alpha_m[i_10] * 1e6, CP=man.cp_m[i_10])
    hz = m.lithologies.shorttle.harzburgite(rhos=man.rho_m[i_10] * 1e-3, alphas=man.alpha_m[i_10] * 1e6, CP=man.cp_m[i_10])

    hlz = m.hydrousLithology(lz, 0.1, continuous=True,
                             phi=0.5)  # continuous melting - only matters for hydrous melting because water is extracted
    hlz_batch = m.hydrousLithology(lz, 0.1)

    # put into pymelt mantle object - mostly lherzolite mantle
    mantle = m.mantle([hlz, px, hz], [6, 2, 2], ['HLz', 'Px', 'Hz'])

    print('pymelt object', mantle.bulkProperties())
    print('T', mantle.adiabat(10, Tp_C) + 273.15, 'K')

    # plot adiabat
    p_plot = np.linspace(0, 22)
    ax.plot(mantle.adiabat(p_plot, Tp_C) + 273.15, p_plot, c='r', label='adiabat')   # inconsistent thermodynamic parameters?
    ax.legend()
    ax.set_xlabel('T (K)')
    ax.set_ylabel('P (GPa)')
    ax.invert_yaxis()
    ax.set_ylim(22, 0)

    # column = mantle.adiabaticMelt(Tp_C)
    # f, a = column.plot()

    plt.show()


# test_solidii(name='Tachinami_buffered2')
decompression_melting(name='Tachinami_buffered2')



