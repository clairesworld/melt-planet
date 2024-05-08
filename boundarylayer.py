import sys
sys.path
sys.path.append('/home/claire/Works/exo-top/')
import exotop.model_1D as blt

planet_kwargs = dict(
    ident='baseline',  # must match dict name ****_in
    M_p=5.972e24,
    sma=1,
    Alb=0,
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
    eta_pre=1.6e11,  # arrhenius pre-exponential factor - dry olivine
    x_Eu=1,  # at 0.7 this is similar to older Treatise values

    #     # viscosity
        a_rh=2.44, # for beta=1/3 from Thiriet+ (2019)
    eta_0=1e21,  # reference eta from Thiriet+ (2019)
    T_ref=1600,  # reference T from Thiriet+ (2019)
    Ea=300e3, # activation energy in J, K&W (1993) dry olivine
    #     V_rh=6e-6, # activation volume in m^3, K&W (1993)  dry olivine
)
run_kwargs = dict(T_m0=1750, T_c0=2250, D_l0=150e3, tf=4.5, visc_type='Thi')  # model params

pl = blt.evolve.build_planet(planet_kwargs, run_kwargs, solve_ODE=True)