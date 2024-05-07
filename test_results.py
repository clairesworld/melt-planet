import matplotlib.pyplot as plt
import numpy as np
from MLTMantle import years2sec, get_Mantle_struct
import PlanetInterior as planet
import sys
sys.path
sys.path.append('/home/claire/Works/rocky-water/py/')
from useful_and_bespoke import colorize

name = 'Tachinami_viscosity'


def plot_pickle_timesteps(name, plot_key, output_path='output/tests/tmp/', xvar=None, fig=None, ax=None, **plot_kwargs):
    import pickle as pkl
    import glob

    # tmp pickle files
    files = glob.glob(output_path + name + '.pkl*')
    log = False

    print(files)

    if plot_key == 'eta':
        log = True

    if fig is None:
        fig = plt.figure()
        ax = plt.gca()

    c = colorize(files, cmap='magma')[0]
    for ii, f in enumerate(files):  # all timesteps
        with open(f, "rb") as pfile:
            d = pkl.load(pfile)
            data = d[plot_key]
            if log:
                data = np.log10(data)
            if xvar is not None:
                ax.plot(xvar, data, c=c[ii], **plot_kwargs)
            else:
                ax.plot(data, c=c[ii], **plot_kwargs)
    ax.set_ylabel(plot_key)
    return fig, ax


# fig, axes = plt.subplots(1, 3)

man = get_Mantle_struct()
# pressures = man.P * 1e-9
#
plot_pickle_timesteps(name, plot_key='u', output_path='output/tests/tmp/',
                      #xvar=pressures, fig=None, ax=None
                      )
plt.show()


""" checks """

# compare viscosity profiles
#
# eta0 = eta_Ranalli(U_0, pressures, **eta_kwargs)
# plt.plot(np.log10(eta0), pressures * 1e-9)
# plt.plot(np.log10(Arrhenius_viscosity_law_pressure(U_0, pressures)), pressures * 1e-9)
# plt.gca().invert_yaxis()
# plt.show()


# viscosity at fixed T_cmb
# eta_b =  alpha * rho ** 2 * gravity * H * L ** 5 / (kappa * kc * RaH)
# eta_b = Arrhenius_viscosity_law(Tcmb0, None, **eta_kwargs_Arr)
# print('eta_b', eta_b)


# density profile
# pl = planet.loadinterior('output/tests/Tachinami_struct.pkl')
# pl.plot_structure_r()
# plt.show()