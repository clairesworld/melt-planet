import PlanetInterior as planet
import MLTMantle as mlt
from HeatTransferSolver import test_arrhenius_radheating, test_pdependence, rad_heating_forward, initial_steadystate, dudx_ambient
import matplotlib.pyplot as plt

name = 'Tachinami_buffered2'

# initialise planetary interior with constant density, alpha, etc
pl = planet.loadinterior('output/tests/Tachinami_struct.pkl')
# pl = planet.PlanetInterior(name=name)
# pl.initialise_constant(n=50000, rho=4500, cp=1190, alpha=3e-5)
# pl.solve()  # solve EoS for depth-dependent thermodynamic parameters
# pl.save(output_path='output/tests/')
# pl.plot_structure_p()

# generate mantle object
man = mlt.MLTMantle(pl, Nm=10000, verbose=False)
# set fixed T boundary conditions/initial condition
man.set_dimensional_attr({'Tsurf': 300, 'Tcmb0': 3000})

# T = initial_steadystate(man.zp * man.d + man.r[0], (man.zp[1] - man.zp[0]) * man.d, man.Tsurf,
#                         man.get_mixing_length_and_gradient(man.zp, **{'alpha_mlt': 0.82, 'beta_mlt': 1})[0],
#                         man.rho_m, man.alpha_m, man.cp_m, man.k_m, man.g_m, man.P, man.Tsurf, man.Tcmb0,
#                         eta_function=mlt.Arrhenius_viscosity_law_pressure, eta_kwargs={},
#                         g_function=rad_heating_forward, g_kwargs={'rho': man.rho_m}, dudx_ambient_function=dudx_ambient
#                         )
# print('T', T)
# plt.plot(T, man.zp * man.d + man.r[0])
# plt.show()


soln = man.solve(t0_Gyr=0, tf_Gyr=5, t0_buffer_Gyr=0,#5,
                 Nt_min=1000,
                 viscosity_law=mlt.Arrhenius_viscosity_law_pressure,
                 internal_heating_function=rad_heating_forward,
                 mixing_length_kwargs={'alpha_mlt': 0.82, 'beta_mlt': 1},  # Tachinami 2011},
                 viscosity_kwargs=None,
                 internal_heating_kwargs={'rad_factor': 1},
                 show_progress=True, verbose=True,
                 writefile='output/tests/' + name + '.h5py')

man.plot_temperature_evol(soln, timestep='all', cmap='magma', figpath='figs_scratch/' + name + '.pdf')

# # test_arrhenius_radheating(N=1000, Nt_min=1000, t_buffer_Myr=3000, age_Gyr=4.5,
# #                           writefile='output/tests/radheating_fixedflux.h5py', plot=True,
# #                           figpath='figs_scratch/radheating_fixedflux.pdf')
#
# test_pdependence(Nt_min=1000, t_buffer_Myr=0, age_Gyr=5, Mantle=man,
#                  writefile='output/tests/' + name + '.h5py', plot=True,
#                  figpath='figs_scratch/' + name + '.pdf')
