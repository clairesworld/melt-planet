""" solve advection-diffusion PDE non steady state
rho * cp * du/dt = -div(q(x,t)) + g(x,t) where q is sum of heat fluxes du/dx, g is source term

"""

import numpy as np
import time
# import sys
import h5py

years2sec = 3.154e7


#################### boundary conditions #########################


def initial(z, u1, u0):
    # initial linear u profile where u0: bottom temperature and u1: top temperature
    return (u1 - u0) * z + u0


def du0dx(t):
    # bottom boundary condition if on flux ∂u/∂x | (x=0, t) (here x=0, but originally ∂u/∂x | (x=L, t)
    return 0


####################### heat transfer functions ###################

def dudx_ambient_constants(u, x, alpha, cp, gravity):
    """ "ambient" T profile at dimensionless z where alpha, cp, gravity are scalars"""
    return -alpha / cp * gravity * u


def dudx_ambient_vectors(u, x, alpha, cp, gravity):
    """ "ambient" T profile at dimensionless x (scalar) where alpha, cp, gravity are vectors"""
    try:
        # return entire profile
        assert np.size(alpha) == np.size(u)
        return -alpha / cp * gravity * u
    except AssertionError:
        # return value at single depth
        assert np.size(x) == 1
        n = int(np.round(x * (len(alpha) - 1)))  # get index -- z from 0 to 1 --- confirmed this works for all n
        return -alpha[n] / cp[n] * gravity[n] * u


def convective_coefficient(alpha, rho, cp, gravity, l, eta, dudx_adiabat, dudx):
    # can't be negative
    return np.maximum(alpha * rho ** 2 * cp * gravity * l ** 4 / (18 * eta) * (dudx_adiabat - dudx), 0)


def radiogenic_heating(t, x, H0=3.4e-11, rho=None, t0_buffer_Gyr=0, **kwargs):
    """Calculate radiogenic heating in W kg^-1 from Korenaga (2006)"""
    sec2Gyr = 1 / 3.154e7 * 1e-9
    t_Gyr = t * sec2Gyr + t0_buffer_Gyr

    # order of isotopes: 238U, 235U, 232Th, 40K
    c_n = np.array([0.9927, 0.0072, 4.0, 1.6256])  # relative concentrations of radiogenic isotopes
    p_n = np.array([9.37e-5, 5.69e-4, 2.69e-5, 2.79e-5])  # heat generation rates (W/kg)
    lambda_n = np.array([0.155, 0.985, 0.0495, 0.555])  # half lives (1/Gyr)
    h_n = (c_n * p_n) / np.sum(c_n * p_n)  # heat produced per kg of isotope normalized to total U
    # print('H0', H0, 'h_n', h_n, 'lambda_n', lambda_n, 't_Gyr', t_Gyr, 'exp', sum(h_n * np.exp(lambda_n * t_Gyr)) )
    h = H0 * sum(h_n * np.exp(-lambda_n * t_Gyr))
    # print('h', h, 'h*rho', h*rho, 'rho', rho)
    return h * rho


def internal_heating_decaying(t, x, age_Gyr=None, x_Eu=1, rho=None, t0_buffer_Gyr=0):
    """ radiogenic heating in W/kg after Table 1 and eqn 1 in O'Neill+ 2020 (SSR)
    x_Eu: concentration of r-process elements wrt solar (i.e. Eu, U, Th)"""
    # order of isotopes (IMPORTANT) is [40K, 238U, 235U, 232Th]
    tau_i = np.array([1250, 4468, 703.8, 14050])  # half life in Myr
    h_i = np.array([28.761e-6, 94.946e-6, 568.402e-6, 26.368e-6])  # heat production in W/kg
    c_i = np.array([30.4e-9, 22.7e-9, 0.16e-9, 85e-9])  # BSE concentration in kg/kg

    sec2Gyr = 1 / 3.154e7 * 1e-9

    # convert times to Myr to be consistent with units of tau
    # add buffer to start simulation from x Gyr (solver has problems if t0>0 ?)
    t_Myr = (t * sec2Gyr + t0_buffer_Gyr) * 1e3
    h_K = np.array(c_i[0] * h_i[0] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[0]))
    try:
        h_UTh = np.array(sum(c_i[1:] * h_i[1:] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[1:])))
    except ValueError:
        t_Myr = np.vstack((t_Myr, t_Myr, t_Myr))
        c_i = c_i.reshape((4, 1))
        h_i = h_i.reshape((4, 1))
        tau_i = tau_i.reshape((4, 1))
        h_UTh = np.array(np.sum(c_i[1:] * h_i[1:] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[1:]), axis=0))

    h_perkg = (h_K + x_Eu * h_UTh)
    # print(t * sec2Gyr * 1e3, 'Myr', 'h', h_perkg, 'W/kg')
    return h_perkg * rho


def internal_heating_constant(t, x, H0=1e-12, **kwargs):
    return H0  # W kg-1, for testing


######################## solve ode in t ###########################


def solve_pde(t0, tf, U_0, heating_rate_function, ivp_args, max_step=1e6 * years2sec,
              verbose=False, show_progress=False, writefile=None):
    """
    :param show_progress:
    :type show_progress:
    :param max_step:
    :type max_step:
    :param t0: start time in seconds
    :type t0: float
    :param tf: end time in seconds
    :type tf: float
    :param U_0:
    :type U_0:
    :param heating_rate_function:
    :type heating_rate_function:
    :param ivp_args:
    :type ivp_args:
    :param verbose:
    :type verbose:
    :return:
    :rtype:
    """
    from scipy.integrate import solve_ivp
    from inspect import signature
    # from tqdm import tqdm

    # if verbose:
    #     # check function signature of heating rate function matches ivp_args?
    #     # todo create ivp_args tuple from kwargs?
    #     sig = signature(heating_rate_function)
    #     print('\nrequired signature for ivp_args:', str(sig))  # skipping t, u
    #     print('ivp_args', ivp_args)
    #     # print('signature of heating_rate_function', list(sig.parameters.keys())[2:])  # skipping t, u

    tspan = tf-t0
    ivp_args = ivp_args + (tspan, show_progress)
    start = time.time()
    soln = solve_ivp(heating_rate_function,
                     t_span=(t0, tf), y0=U_0,
                     method='BDF',  # conserves energy
                     # method='LSODA',
                     t_eval=None,
                     vectorized=False,
                     args=ivp_args,
                     max_step=max_step,
                     jac=None  # recommended for BDF, todo
                     )
    end = time.time()

    if writefile is not None:
        # write output to h5py file
        with h5py.File(writefile, "w") as file:
            file.create_dataset('temperature', data=soln.y, dtype=soln.y.dtype)
            file.create_dataset('time', data=soln.t, dtype=soln.t.dtype)
            # file.create_dataset('z', data=np.linspace(0, 1, len(soln.y)), dtype=np.float64)

    print('\n')
    if verbose:
        print(len(soln.t), 'timesteps in', end - start, 'seconds', '(' + heating_rate_function.__name__ + ')')
        print('    u', np.shape(soln.y), ', t', np.shape(soln.t))
    return soln


################### different methods to calculate total heating rate on RHS ###########################


def calc_total_heating_rate_numeric(t, u, dx, xprime, l_function, dudx_ambient_function, eta_function, g_function, kc,
                                    alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l,
                                    tspan, show_progress):
    """ function to calculate dT/dt for each z (evaluated at array of temperatures u)
    this can be sped up by ~2 if mixing length is time-independent"""
    if show_progress:
        print("\rModel time: " + str(format(t/years2sec, ".0f")) + " yr, Completion percentage " + str(format(((t / tspan) * 100), ".4f")) + "%",
              end='', flush=True)

    # update viscosity
    eta = eta_function(u, xprime, **eta_kwargs)
    # print('viscosity range', eta[0], eta[-1], 'Pa s')
    # print('eta', eta)
    # print('T range', u[0], u[-1], 'K')

    if l is None:
        # update mixing length
        lp, _ = l_function(xprime, **l_kwargs)
        l = lp * L  # dimensionalise

    # analytic derivatives
    dudx_adiabat = dudx_ambient_function(u, xprime, alpha, cp, gravity)

    # discretized derivatives
    dudx = np.gradient(u, dx)

    # convective heat transport
    kv = convective_coefficient(alpha, rho, cp, gravity, l, eta, dudx_adiabat, dudx)

    source_term = g_function(t, xprime, **g_kwargs)
    # print('internal heating', source_term, 'W/m3' )

    # calculate divergence of flux
    diff_term = -kc * dudx
    adv_term = -kv * (dudx - dudx_adiabat)
    q = diff_term + adv_term
    divq = np.gradient(q, dx)

    lhs = -divq + source_term

    # redo boundary conditions (np.gradient might do this differently)
    lhs[-1] = 0  # value of du/dt at x=L - i.e. constant temperature so no dT/dt

    # # # bottom bdy condition using constant flux - can assume diffusion only
    # lhs[0] = (kc/dx**2)*(2*u[1] + 2*dx*du0dx(t) - 2*u[0]) + source_term

    # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
    lhs[0] = 0  # value of du/dt at x=L - i.e. constant temperature so no dT/dt

    dudt = lhs / (rho * cp)

    return dudt


def calc_total_heating_rate_analytic_isoviscous(t, u, dx, dudx_ambient_function, g_function, l, dldx, kc,
                                                alpha, rho, cp, gravity, eta, g_kwargs):
    """ # function to calculate dT/dt for each z (evaluated at array of temperatures u).
     like the above but as analytic as possible
     for isoviscous case this is  ~0.1 sec faster
     """
    N = len(u)
    rhs = np.zeros(N)
    rhs[-1] = 0  # surface boundary condution, dT/dt = 0 because constant T

    source_term = g_function(t, None, **g_kwargs)  # assuming no depth-dependence

    # isoviscous case can be done analytically
    A = alpha * rho ** 2 * cp * gravity * l ** 4 / (18 * eta)
    dAdx = 4 * alpha * rho ** 2 * cp * gravity * l ** 3 / (18 * eta) * dldx

    # building profile from top
    for i in range(N - 2, 0, -1):

        # discretize derivatives - central differences
        d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)  # second order finite difference
        dudx = (u[i + 1] - u[i - 1]) / (2 * dx)  # central difference
        dudx_adiabat = dudx_ambient_function(u[i], None, alpha, cp, gravity)
        d2udx2_adiabat = -alpha / cp * gravity * dudx  # temp - assuming virtually constant alpha, cp, g

        # check for convection
        if abs(dudx_adiabat) > abs(dudx):  # -ve kv, no convection
            rhs[i] = kc * d2udx2 + source_term
        #             print('kv = 0 at z =', i)
        else:
            rhs[i] = kc * d2udx2 - dAdx[i] * (dudx ** 2 - 2 * dudx * dudx_adiabat + dudx_adiabat ** 2) \
                     - 2 * A[i] * (
                             dudx * d2udx2 - d2udx2 * dudx_adiabat - dudx * d2udx2_adiabat + dudx_adiabat * d2udx2_adiabat) \
                     + source_term

    #     # bottom bdy condition using constant flux - can assume diffusion only
    #     rhs[0] = (kc/dx**2)*(2*u[1] + 2*dx*du0dx(t) - 2*u[0]) + source_term

    # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
    rhs[0] = 0

    return rhs / (rho * cp)


# function to calculate dT/dt for each z (evaluated at array of temperatures u)
# like the above but as analytic as possible
def calc_total_heating_rate_analytic(t, u, dx, xprime, l_function, dudx_ambient_function, eta_function, g_function, kc,
                                     alpha, rho, cp,
                                     gravity, L, l_kwargs, eta_kwargs, g_kwargs):
    N = len(u)
    rhs = np.zeros(N)
    rhs[-1] = 0  # surface boundary condution, dT/dt = 0 because constant T

    source_term = g_function(t, xprime, **g_kwargs)

    # update eta(T)
    eta = eta_function(u, xprime, **eta_kwargs)

    # isoviscous case can be done analytically
    lp, dldx = l_function(xprime, **l_kwargs)
    l = lp * L  # dimensionalise
    f = alpha * rho ** 2 * cp * gravity / (18 * eta)
    df = np.gradient(f, dx)  # a bit too complicated to do analytically
    g = l ** 4
    dg = 4 * l ** 3 * dldx

    A = f * g

    try:
        dAdx = (f * dg) + (df * g)
    except ValueError:
        assert len(df) == 0  # f is a scalar, df is here an empty list (actually =0)
        dAdx = f * dg

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(zp, A)
    # ax[1].plot(zp, dAdx)
    # ax[0].set_ylabel('A')
    # ax[1].set_ylabel('dAdx')
    # ax[0].legend(['t={:.3f} Myr'.format(t / years2sec * 1e-6)])
    # plt.show()

    # building profile from top
    for i in range(N - 2, 0, -1):

        # discretize derivatives - central differences
        d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)  # second order finite difference
        dudx = (u[i + 1] - u[i - 1]) / (2 * dx)  # central difference
        dudx_adiabat = dudx_ambient_function(u[i], xprime, alpha, cp, gravity)
        d2udx2_adiabat = -alpha / cp * gravity * dudx  # temp - assuming virtually constant alpha, cp, g

        # check for convection
        if abs(dudx_adiabat) > abs(dudx):  # -ve kv, no convection
            rhs[i] = kc * d2udx2 + source_term
        #             print('kv = 0 at z =', i)
        else:
            rhs[i] = kc * d2udx2 - dAdx[i] * (dudx ** 2 - 2 * dudx * dudx_adiabat + dudx_adiabat ** 2) \
                     - 2 * A[i] * (
                             dudx * d2udx2 - d2udx2 * dudx_adiabat - dudx * d2udx2_adiabat + dudx_adiabat * d2udx2_adiabat) \
                     + source_term

    #     # bottom bdy condition using constant flux - can assume diffusion only
    #     rhs[0] = (kc/dx**2)*(2*u[1] + 2*dx*du0dx(t) - 2*u[0]) + source_term

    # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
    rhs[0] = 0

    return rhs / (rho * cp)


############################### tests #############################################

def test_isoviscous(N=500, Nt_min=0, writefile=None, verbose=True, plot=True):
    """ test generic case """
    from MLTMantle import get_mixing_length_and_gradient_smooth, Arrhenius_viscosity_law
    import matplotlib.pyplot as plt
    import PlanetInterior as planet
    import melting_functions as melt

    def internal_heating_constant(t, x, **kwargs):
        return 1e-12  # W kg-1, for testing

    def viscosity_constant(u, x, **kwargs):
        return 10 ** 21

    # set up grid
    L = 3000e3  # length scale
    D = 1  # dimensionless length scale
    zp = np.linspace(0, 1, N)  # dimensionless height

    # increments and domain
    dx = (zp[1] - zp[0]) * L
    t0, tf = 0, 5e9 * years2sec

    try:
        max_step = (tf-t0)/Nt_min
        if verbose:
            print('max step:', max_step / years2sec, 'years')
    except ZeroDivisionError:
        max_step = np.inf
        if verbose:
            print('max step: inf')

    # constants
    alpha_mlt = 0.2895
    beta_mlt = 0.6794
    lp, dldx = get_mixing_length_and_gradient_smooth(zp, alpha_mlt, beta_mlt)
    l = lp * L
    cp = 1190  # J/kg/K
    alpha = 3e-5
    gravity = 10
    kc = 5
    rho = 4500  # kg/m3
    H = 1e-12  # W/kg
    RaH = 1e7
    dEta = 1e6
    kappa = kc / (rho * cp)

    l_kwargs = {'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt}
    g_kwargs = {'rho': rho, 'H': H}  # not relevant here but args passed to g_function
    # eta_kwargs = {'RaH': RaH, 'dEta': dEta, 'kappa': kappa}
    eta_kwargs = {'eta_ref': 1e21, 'T_ref': 1600, 'Ea': 300e3}

    Tsurf = 300
    Tcmb0 = 2850

    U_0 = initial(zp, Tsurf, Tcmb0)  # initial temperature

    ivp_args2 = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient_constants, viscosity_constant,
                 internal_heating_constant, kc, alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l)
    soln2 = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args2, verbose=verbose, show_progress=True,
                      writefile=writefile, max_step=max_step)

    # ivp_args1 = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient_constants, Arrhenius_viscosity_law,
    #              internal_heating_constant, kc, alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l)
    # soln1 = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args1, verbose=verbose, show_progress=True,
    #                       writefile=writefile, max_step=max_step)

    # pl = planet.PlanetInterior()
    # pl.initialise_constant(n=5000, rho=4500, cp=1190, alpha=3e-5)
    # # pl.solve()
    # Tsol = [melt.T_solidus_pyrolite(pp) for pp in pl.pressure[pl.i_cmb + 1:]]
    # zt = np.linspace(pl.radius[pl.i_cmb + 1:], pl.radius[-1], len(Tsol))

    if plot:
        # plot
        n = len(soln2.t) - 1
        plt.figure()
        # plt.plot(zp, soln.y[:, n], label='analytic')
        # plt.plot(zp, soln1.y[:, n], label='numeric Arrhenius')
        plt.plot(zp, soln2.y[:, n], label='numeric isoviscous')
        # plt.plot(zt, Tsol, label='pyrolite solidus')
        plt.xlabel('z/L')
        plt.ylabel('T (K)')
        plt.title('t={:.3f} Myr'.format(soln2.t[n] / years2sec * 1e-6))
        plt.legend()
        # i = N - 1
        # plt.figure()
        # plt.plot(soln.t / years2sec * 1e-6, soln.y[i, :])
        # plt.xlabel('time (Myr)')
        # plt.ylabel('Surface temperature (K)')
        plt.show()


def test_viscositycontrast(N=1000, Nt_min=1000, verbose=True, writefile=None, plot=True):
    """ test generic case """
    from MLTMantle import get_mixing_length_and_gradient_smooth, exponential_viscosity_law, Arrhenius_viscosity_law
    from MLTMantleCalibrated import get_mixing_length_calibration
    import matplotlib.pyplot as plt

    # set up grid/domain
    L = 3000e3  # length scale
    D = 1  # dimensionless length scale
    zp = np.linspace(0, 1, N)  # dimensionless height
    dx = (zp[1] - zp[0]) * L
    t0, tf = 0, 5e9 * years2sec  # seconds

    try:
        max_step = (tf-t0)/Nt_min
        if verbose:
            print('max step:', max_step / years2sec, 'years')
    except ZeroDivisionError:
        max_step = np.inf
        if verbose:
            print('max step: inf')

    # dimensionless convective parameters
    RaH = 1e7
    dEta = 1e4
    # mixing length calibration (stagnant lid mixed heated)
    # alpha_mlt = 0.2895
    # beta_mlt = 0.6794
    alpha_mlt, beta_mlt = get_mixing_length_calibration(RaH, dEta)
    if verbose:
        print('alpha_mlt', alpha_mlt, 'beta_mlt', beta_mlt)

    # constants
    lp, dldx = get_mixing_length_and_gradient_smooth(zp, alpha_mlt, beta_mlt)
    l = lp * L
    cp = 1190  # J/kg/K
    alpha = 3e-5
    gravity = 10
    kc = 5
    rho = 4500  # kg/m3
    H0 = 3.4e-11  # 1e-12  # W/kg
    kappa = kc / (rho * cp)

    # boundary conditions
    Tsurf = 300
    Tcmb0 = 2850
    U_0 = initial(zp, Tsurf, Tcmb0)  # initial temperature

    l_kwargs = {'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt}
    g_kwargs_constant = {'rho': rho, 'H': H0}  # not relevant here but args passed to g_function
    g_kwargs_decay = {'H0': H0, 'rho': rho, 't0_buffer_Gyr': 0}
    eta_kwargs_Arr = {'eta_ref': 1e21, 'T_ref': 1600, 'Ea': 300e3}

    # # plot internal heating
    # plt.figure()
    # t = np.linspace(t0, tf)  # seconds
    # print('tf', t[-1], 'seconds')
    # plt.plot(t / (years2sec * 1e9), [radiogenic_heating(tt, x=None, **g_kwargs_decay) for tt in t],
    #          label='Radiogenic heating')
    # plt.xlabel('t (Gyr)')
    # plt.ylabel('H (W/m3)')
    # plt.legend()
    # plt.show()

    # viscosity at fixed T_cmb
    # eta_b =  alpha * rho ** 2 * gravity * H * L ** 5 / (kappa * kc * RaH)
    eta_b = Arrhenius_viscosity_law(Tcmb0, None, **eta_kwargs_Arr)
    print('eta_b', eta_b)

    ivp_args = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient_constants, Arrhenius_viscosity_law,
                radiogenic_heating, kc, alpha, rho, cp, gravity, L,
                l_kwargs, eta_kwargs_Arr, g_kwargs_decay, l)  # needs to match signature to heating_rate_function
    soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args,
                     verbose=True, show_progress=True, max_step=max_step, writefile=writefile)

    # eta_kwargs2 = {'Tsurf': Tsurf, 'Tcmb': Tcmb0, 'dEta': dEta, 'eta_b': eta_b}
    # ivp_args2 = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient_constants, exponential_viscosity_law,
    #              internal_heating_constant, kc, alpha, rho, cp, gravity, L,
    #              l_kwargs, eta_kwargs2, g_kwargs, l)  # needs to match signature to heating_rate_function
    # soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args2, verbose=True, writefile=writefile)
    #
    # ivp_args3 = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient_constants, Arrhenius_viscosity_law,
    #             internal_heating_constant, kc, alpha, rho, cp, gravity, L,
    #             l_kwargs, eta_kwargs_Arr, g_kwargs_constant, l)  # needs to match signature to heating_rate_function
    # soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args3,
    #                  verbose=verbose, show_progress=True, max_step=max_step, writefile=writefile)

    if plot:
        # plot
        n = -1
        plt.figure()
        plt.plot(zp, soln.y[:, n], label='Arrhenius')
        # plt.plot(zp, soln2.y[:, n], label='exponential')
        plt.xlabel('z/L')
        plt.ylabel('T (K)')
        plt.title('t={:.3f} Myr'.format(soln.t[n] / years2sec * 1e-6))
        plt.legend()

        # i = N - 1
        # plt.figure()
        # plt.plot(soln.t / years2sec * 1e-6, soln.y[i, :])
        # plt.xlabel('time (Myr)')
        # plt.ylabel('Surface temperature (K)')

        n = 0  # initial
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(zp, np.log10(Arrhenius_viscosity_law(soln.y[:, n], zp, **eta_kwargs_Arr)), label='Arrhenius')
        # ax[0].plot(zp, np.log10(
        #     exponential_viscosity_law(soln2.y[:, n], zp, **eta_kwargs2)), label='exponential')
        # ax[1].plot(zp, kv)
        ax[0].set_ylabel(r'log$\eta$')
        # ax[1].set_ylabel('kv')
        fig.suptitle('t={:.3f} Myr'.format(soln.t[n] / years2sec * 1e-6))

        plt.show()


# test_isoviscous(writefile='isoviscous.h5py')
test_viscositycontrast(N=1000, Nt_min=1000, writefile='radheating.h5py')