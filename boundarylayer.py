import MLTMantle as mlt
import sys
import numpy as np
import matplotlib.pyplot as plt
import model_1D as blt
from model_1D import evolve, thermal, the_results
M_E = 5.972e24  # earth mass in kg
# sys.path
# sys.path.append('/home/claire/Works/exo-top/')



planet_kwargs = dict(
    ident='baseline',  # must match dict name ****_in
    M_p=M_E,
    CMF=0.3,
    Ra_crit_u=450,
    rho_m=3500,
    rho_c=7200,
    c_m=1142,
    c_c=840,
    beta_u=1 / 3,
    k_m=4,
    alpha_m=2.5e-5,
    T_s=273,
    x_Eu=1,  # at 0.7 this is similar to older Treatise values

    #     # viscosity
    a_rh=2.44, # for beta=1/3 from Thiriet+ (2019)
    eta_0=10 ** 21.65295658212541,  # reference eta from Arrhenius law at 4 GPa
    T_ref=1600,  # reference T from Thiriet+ (2019)
    Ea=300e3, # activation energy in J, K&W (1993) dry olivine
    #     V_rh=6e-6, # activation volume in m^3, K&W (1993)  dry olivine
)
run_kwargs = dict(T_m0=1750, T_c0=2250, D_l0=150e3, tf=4.5, visc_type='Thi')  # model params

# print(np.log10(mlt.Arrhenius_viscosity_law_pressure(1600, 4e9)))

man = mlt.get_Mantle_struct(Nm=1000)  # load interior structure
P_GPa = man.P * 1e-9


def potential_temperature(T, P, alpha, rho, cp):
    return T / np.exp(alpha/(rho * cp) * P)

def convective_profile(Tm, pl, man):
    Tp = potential_temperature(Tm, man.P, pl.alpha_m, pl.rho_m, pl.c_m)

    # Tconv = (Tp * np.exp(pl.alpha_m / (pl.rho_m * pl.c_m) * man.P))

    # Seales, Lenardic+ 2022
    Tconv = Tm * (1 - (pl.g_sfc * pl.alpha_m / pl.c_m) * (pl.R_p - pl.R_c)/2 - man.r)
    return Tconv

def T_profile(pl, man, lid_heating=False, t_idx=-1):
    a0_lid = 0
    lid_base = (np.abs(man.r - pl.R_l[t_idx])).argmin()
    lid_radii = man.r[lid_base:]
    T_cond = thermal.sph_conduction(lid_radii, a0=a0_lid, T_l=pl.T_l[t_idx], R_l=pl.R_l[t_idx], k_m=pl.k_m,
                                          T_s=pl.T_s, R_p=pl.R_p)

    Tm = pl.T_m[t_idx]  # temperature at base of lithosphere
    print('Tm', Tm)
    T_conv = convective_profile(Tm, pl, man)

    fig = plt.figure()
    plt.plot(T_cond, man.P[lid_base:] * 1e-9, label='lithosphere')
    plt.plot(T_conv[:lid_base], man.P[:lid_base] * 1e-9, label='mantle')
    plt.xlabel('T (K)')
    plt.ylabel('P (GPa)')
    plt.legend()
    plt.gca().invert_yaxis()



pl = evolve.build_planet(planet_kwargs, run_kwargs, solve_ODE=True)

fig, ax = plt.subplots(1, 1)
ax.plot(pl.t * 1e-9 / mlt.years2sec, pl.T_m)
ax.set_xlabel('t')
ax.set_ylabel('T')

T_profile(pl, man, lid_heating=False, t_idx=-1)

plt.show()