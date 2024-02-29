import time
import PlanetInterior as planet
import MLTMantleCalibrated as mlt
import matplotlib.pyplot as plt
import numpy as np

start = time.time()

# initialise planetary interior with constant density, alpha, etc
pl = planet.PlanetInterior()
pl.initialise_constant(n=50000, rho=4500, cp=1190, alpha=3e-5)

# # plot up structure - OK
# pl.plot_structure_r(save=False)

# generate mantle object
man = mlt.MLTMantleCalibrated(pl, Nm=10000, verbose=False)

# solve steady state temperature
RaH = 1e7  # Rayleigh number for internal heating
dEta = 1e4  # viscosity contrast
H = 1e-12  # W kg-1
Tb = 2850  # this actually gives a reasonable approximation given adiabatic profile
Tsurf = 300

# set fixed T boundary conditions
man.set_dimensional_attr({'Tsurf': Tsurf, 'Tb': Tb})

""" checks """

# check mixing length - OK
l = np.array([man.get_mixing_length(z, Ra_b=None, dEta=None) * man.d for z in man.zp])  # dimensional mixing length
# plt.figure()
# plt.plot(man.zp, [ll * 1e-3 for ll in l], c='r')
# plt.ylabel(r'$l$ (km)')
# plt.xlabel(r'$r^\prime$')

# # check ambient dTdz
# gamma = [man.get_dTdz_ambient(z)*1e3 for z in man.zp]  # adiabatic lapse rate in K/km

# double check adiabatic T(z) - OK
T_ad = np.zeros(man.Nm)
T_ad[-1] = man.planet.Tp
dr = man.m * man.d * 1e-3  # km
for i in range(2, man.Nm + 1):
    lapse_rate = gamma[man.Nm - i]
    T_ad[man.Nm - i] = T_ad[man.Nm - i + 1] - dr * lapse_rate
# fig, axes = plt.subplots(1, 2)
# axes[0].plot(man.zp, gamma)  # should be about 0.3-0.5 K/km for Earth
# axes[0].set_ylabel(r'$dT/dz$ (K/km)')
# axes[0].set_xlabel(r'$r^\prime$')
# axes[1].plot(man.zp, T_ad)
# axes[1].plot((man.planet.radius[man.planet.i_cmb:] - man.planet.Rc) / man.d,
#              man.planet.temperature[man.planet.i_cmb:], ls='--')
# axes[1].set_ylabel(r'adiabatic temperature (K)')
# axes[1].set_xlabel(r'$r^\prime$')

# check viscosity profile along adiabat - OK
Tprime = (T_ad - T_ad[-1]) / abs((T_ad[-1] - T_ad[0]))  # for fixed temperature contrast
print("T'b", Tprime[0], "T's", Tprime[-1])
eta = man.get_viscosity(T_ad, P=None, RaH=RaH, dEta=dEta, H=H)
# fig, ax = plt.subplots(1, 1)
# ax.plot(man.zp, np.log10(eta))
# ax.set_ylabel(r'log$\eta$')
# ax.set_xlabel(r'$r^\prime$')

# calculate steady state mantle temperature
T = man.build_steadystate_Tz(RaH=RaH, dEta=dEta, H=H)
plt.figure()
plt.plot(man.zp, T, c='r', label='avg profile')
plt.ylabel(r'$T$')
plt.xlabel(r'$r^\prime$')
plt.title('logRaH = ' + str(np.log10(RaH)) + ', log$\Delta\eta$ = ' + str(np.log10(dEta)))

# kind of works except would need to rescale to fixed boundary condition... I think you do need both boundary conditions
# instead of assuming C = 0 in steady state -- this is where the other one would come in... but ignore for now to
# move on to time-dependent solution...

# check typical chi values
chi = man.alpha_m * man.rho_m ** 2 * man.cp_m * man.g_m * l ** 4 / (18 * eta)
plt.figure()
plt.plot(man.zp, chi, c='r')
plt.ylabel(r'$\chi$')
plt.xlabel(r'$r^\prime$')

# Calculate the end time and time taken
end = time.time()
print("run time", end - start, "s")

plt.tight_layout()
plt.show()
