import matplotlib.pyplot as plt
import numpy as np
from MLTMantle import years2sec, get_Mantle_struct, get_mixing_length_and_gradient_smooth
import PlanetInterior as planet
from HeatTransferSolver import dudx_ambient
import sys
sys.path
sys.path.append('/home/claire/Works/rocky-water/py/')
from useful_and_bespoke import colorize



def plot_pickle_timesteps(name, plot_key, output_path='output/tests/tmp/', xvar=None, fig=None, ax=None, last=False, **plot_kwargs):
    import pickle as pkl
    import glob

    # tmp pickle files
    files = glob.glob(output_path + name + '.pkl*')
    log = False

    files = sorted(files)

    if plot_key == 'eta':
        log = True

    if fig is None:
        fig = plt.figure()
        ax = plt.gca()

    if last:
        files = [files[-1]]

    c = colorize(np.arange(len(files)), cmap='magma')[0]
    for ii, f in enumerate(files):  # all timesteps
        with open(f, "rb") as pfile:
            d = pkl.load(pfile)
            data = d[plot_key]
            if log:
                data = np.log10(data)
            if xvar is not None:
                try:
                    ax.plot(d[xvar], data, c=c[ii], **plot_kwargs)
                except (KeyError, TypeError):
                    ax.plot(xvar, data, c=c[ii], **plot_kwargs)
            else:
                ax.plot(data, c=c[ii], **plot_kwargs)
    ax.set_ylabel(plot_key)
    return fig, ax

def plot_velocity(man, name, output_path, fig=None, ax=None):
    # need to find base of lithosphere
    import glob
    import pickle as pkl

    # tmp pickle files
    files = glob.glob(output_path + name + '.pkl*')
    files = sorted(files)

    if fig is None:
        fig = plt.figure()
        ax = plt.gca()

    with open(files[-1], "rb") as pfile:
        d = pkl.load(pfile)
        u = d['u']
        eta = d['eta']

    z = np.linspace(0, 1, len(u))
    alpha = man.alpha_m
    rho = man.rho_m
    g = man.g_m
    cp = man.cp_m
    l = get_mixing_length_and_gradient_smooth(z, alpha_mlt=0.82, beta_mlt=1, l_smoothing_distance=0.05)[0]
    dudx_adiabat = dudx_ambient(u, z, alpha, cp, g)

    dudx = np.gradient(u, z)
    dT = (dudx_adiabat - dudx) ** 2
    v = alpha * rho * g * l ** 2 / (18 * eta) * dT

    A = 4 * np.pi * man.r[-1] ** 2 * 0.5  # half of surface area is upwelling
    dmdt = rho * A * v  # mass flux through surface of sphere

    ax.plot(z, v)
    ax.set_ylabel('velocity (m/s)')



name = 'Tachinami_viscosity'

# fig, axes = plt.subplots(1, 3)

man = get_Mantle_struct(Nm=1000)
pressures = man.P * 1e-9
#
# plot_pickle_timesteps(name, plot_key='u', output_path='output/tests/tmp/',
#                       #xvar=pressures, fig=None, ax=None
#                       )

# plot_velocity(man, name, output_path='output/tests/tmp/', fig=None, ax=None)

plot_pickle_timesteps(name, 'eta', output_path='output/tests/tmp/', xvar=pressures, last=True)

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
plt.show()