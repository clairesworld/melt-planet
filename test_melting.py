import time
import PlanetInterior as planet
import MLTMantleCalibrated as mlt
from MLTMantle import exponential_viscosity_law, Arrhenius_viscosity_law, constant_viscosity_law, get_mixing_length_and_gradient_smooth
from HeatTransferSolver import internal_heating_decaying, radiogenic_heating
import matplotlib.pyplot as plt
import numpy as np

years2sec = 3.154e7


def internal_heating_constant(t, x, **kwargs):
    return 1e-12  # W kg-1, for testing

start = time.time()

# initialise planetary interior with depth-dependent density etc
pl = planet.PlanetInterior()
pl.initialise_constant(n=5000, rho=4500, cp=1190, alpha=3e-5)
# pl.solve()

# generate mantle object
man = mlt.MLTMantleCalibrated(pl, Nm=1000, verbose=False)

# define convective parameters and boundary conditions
RaH = 1e7  # Rayleigh number for internal heating
dEta = 1e6  # viscosity contrast
eta_b = 1e17  # not consistent with Ra
Tb = 2850  # this actually gives a reasonable approximation given adiabatic profile
Tsurf = 300

alpha_mlt, beta_mlt = mlt.get_mixing_length_calibration(RaH, dEta)

# set fixed T boundary conditions
man.set_dimensional_attr({'Tsurf': Tsurf, 'Tcmb0': Tb})

# set time domain
t0_Gyr, tf_Gyr = 0, 5
t0_buffer_Gyr = 0

# choose internal heating method
internal_heating_function = internal_heating_constant
internal_heating_kwargs = {}
# internal_heating_function = internal_heating_decaying
# internal_heating_kwargs = {'age_Gyr': tf_Gyr, 'x_Eu': 1, 'rho': man.rho_m[0],
#                            't0_buffer_Gyr': t0_buffer_Gyr}
# internal_heating_kwargs1 = {'H0': 3.4e-11, 'rho': man.rho_m[0],
#                            't0_buffer_Gyr': t0_buffer_Gyr}
#
# # check internal heating
# t = np.linspace(t0_Gyr, tf_Gyr) * years2sec * 1e9
# h = np.array([internal_heating_decaying(tt, x=None, **internal_heating_kwargs) for tt in t])
# h2 = np.array([radiogenic_heating(tt, x=None, **internal_heating_kwargs1) for tt in t])
# plt.plot(t/years2sec * 1e-9, h/man.rho_m[0], label='decaying')
# plt.plot(t/years2sec * 1e-9, h2/man.rho_m[0], label='forward', ls='--')
# plt.legend()
# plt.xlabel('t (Gyr)')
# plt.ylabel('internal heating rate (W/kg)')
# plt.show()

# choose viscosity law
# viscosity_law = exponential_viscosity_law
# viscosity_kwargs = {'dEta': dEta, 'Tsurf': Tsurf, 'Tcmb': Tb, 'eta_b': eta_b}
# viscosity_law = Arrhenius_viscosity_law
# viscosity_kwargs = {'eta_ref': 1e21, 'T_ref': 1600, 'Ea': 300e3}
viscosity_law = constant_viscosity_law
viscosity_kwargs = {'eta0': 1e20}

# # check initial viscosity profile
# from HeatTransferSolver import initial
# T = initial(man.zp, Tsurf, Tb)
# eta = viscosity_law(T, None, **viscosity_kwargs)
# plt.plot(man.zp, np.log10(eta))
# plt.xlabel('z/L')
# plt.ylabel(r'log($\eta$')
# plt.show()

# check mixing lnegth
lp, dldx = get_mixing_length_and_gradient_smooth(man.zp, **{'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt})
l = lp * man.d  # dimensionalise
plt.figure()
plt.plot(man.zp, l * 1e-3)
plt.xlabel('z/L')
plt.ylabel('mixing length (km)')
plt.show()

# solve temperature evolution
soln = man.solve(t0_Gyr, tf_Gyr,  t0_buffer_Gyr=t0_buffer_Gyr,
                 viscosity_law=viscosity_law,
                 internal_heating_function=internal_heating_function,
                 radiogenic_conentration_factor=1,
                 mixing_length_kwargs={'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt},
                 viscosity_kwargs=viscosity_kwargs,
                 internal_heating_kwargs=internal_heating_kwargs,
                 max_step=1e6 * years2sec,  #np.inf, #1e4 * years2sec,
                 show_progress=True, verbose=True,
                 )

# plot
n = -1
plt.figure()
plt.plot(man.zp, soln.y[:, n], label='isoviscous')
plt.xlabel('z/L')
plt.ylabel('T (K)')
plt.title('t={:.3f} Myr'.format(soln.t[n] / years2sec * 1e-6))
plt.legend()
plt.show()
