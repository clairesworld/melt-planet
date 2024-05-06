import matplotlib.pyplot as plt
import numpy as np
from MLTMantle import years2sec, get_Mantle_struct
import PlanetInterior as planet

name = 'Tachinami_full'


def load_pickle_timesteps(name, plot_key, output_path='output/tests/tmp/', xvar=None, fig=None, ax=None):
    import pickle as pkl
    import glob

    # tmp pickle files
    files = glob.glob(output_path + name + '.pkl*')
    log = False

    if plot_key == 'eta':
        log = True

    if fig is None:
        fig = plt.figure()
        ax = plt.gca()

    for f in files:  # all timesteps
        with open(f, "rb") as pfile:
            d = pkl.load(pfile)
            data = d[plot_key]
            if log:
                data = np.log10(data)
            if xvar is not None:
                ax.plot(xvar, data)
            else:
                ax.plot(data)


# fig, axes = plt.subplots(1, 3)

man = get_Mantle_struct()
# pressures = man.P * 1e-9
#
# load_pickle_timesteps(name, plot_key='q', output_path='output/tests/tmp/', xvar=pressures, fig=None, ax=None)
# plt.show()


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