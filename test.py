import PlanetInterior as planet
import MLTMantle as mlt
from HeatTransferSolver import test_arrhenius_radheating, test_pdependence

name = 'Tachinami'

# initialise planetary interior with constant density, alpha, etc
pl = planet.loadinterior('output/tests/Tachinami_struct.pkl')
# pl = planet.PlanetInterior(name=name)
# pl.initialise_constant(n=50000, rho=4500, cp=1190, alpha=3e-5)
# pl.solve()  # solve EoS for depth-dependent thermodynamic parameters
# pl.save(output_path='output/tests/')
# pl.plot_structure_p()

# generate mantle object
man = mlt.MLTMantle(pl, Nm=10000, verbose=False)
# set fixed T boundary conditions
man.set_dimensional_attr({'Tsurf': 300, 'Tcmb0': 2500})


# # test_arrhenius_radheating(N=1000, Nt_min=1000, t_buffer_Myr=3000, age_Gyr=4.5,
# #                           writefile='output/tests/radheating_fixedflux.h5py', plot=True,
# #                           figpath='figs_scratch/radheating_fixedflux.pdf')
#
test_pdependence(Nt_min=1000, t_buffer_Myr=0, age_Gyr=5, Mantle=man,
                 writefile='output/tests/' + name + '.h5py', plot=True,
                 figpath='figs_scratch/' + name + '.pdf')
