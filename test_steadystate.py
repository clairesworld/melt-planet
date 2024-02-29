import PlanetInterior as planet
import MLTMantleSteadystate_VK as mltss
import matplotlib.pyplot as plt

# initialise planetary interior with constant density, alpha, etc
pl = planet.PlanetInterior()
pl.initialise_constant(n=50000, rho=None, cp=1300, alpha=2.5e-5)

# plot up structure
pl.plot_structure_r(save=False)

# generate mantle object
mtl = mltss.MLTMantleSteadystate(pl, Nm=10000, verbose=True)

# calculate steady state mantle temperature

RaH = 1e6
H = 4.87e-8  # for RaH = 1e6 - max melting}

solution = mtl.solve(RaH, H)
z = mtl.zp
Thot = solution['Thot']
Tavg = solution['Tavg']
T95 = solution['T95']

plt.figure()
plt.plot(Thot, z, c='r', label='hot profile')
plt.plot(Tavg, z, c='b', label='average profile')
plt.plot(T95, z, c='orange', label='95th hottest')
plt.legend()
plt.xlabel(r'$T^\prime$')
plt.ylabel(r'$d^\prime$')

plt.show()

"""
works in steady state in the sense VK is reproduced (execept for supersolidus % but this is very sensitive to exact
solidus value). Next steps:
0. email Paul Tackley back!!!!
1. see if you can find steady state solution for Wagner parameterisation in dimensionless form
2. retry time stepping Wagner whilst dimensionless
"""

