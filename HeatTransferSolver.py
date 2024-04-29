""" solve advection-diffusion PDE non steady state
rho * cp * du/dt = -div(q(x,t)) + g(x,t) where q is sum of heat fluxes du/dx, g is source term

"""

import numpy as np
import time
# import sys
import pickle as pkl
import matplotlib.pyplot as plt

years2sec = 3.154e7


#################### boundary conditions #########################


def initial_file(fin, outputpath="/home/claire/Works/melt-planet/output/tests/"):
    from MLTMantle import read_h5py
    soln = read_h5py(fin, outputpath, verbose=True)
    return soln['temperature'][:,-1]


def initial_steadystate(z, Tsurf, Tcmb0, alpha, cp, g, l, rho, kc, pressures, g_function, g_kwargs, eta_function, eta_kwargs,
                        dudx_ambient_function, tol=1e-4):
    # iterative method
    N = len(z)
    dz = z[1] - z[0]
    zp = (z - z[0]) / (z[-1] - z[0])
    T = initial_linear(zp, Tsurf, Tcmb0)  # starting guess

    H = g_function(t=0, x=z, **g_kwargs)

    T0_old = 100000000000
    T0_new = T[0]

    while abs(T0_old - T0_new) > tol:

        for i in range(N - 2, -1, -1):  # from top
            alphai = alpha[i]
            cpi = cp[i]
            gi = g[i]
            li = l[i]
            rhoi = rho[i]
            kci = kc[i]
            Hi = H[i]
            zi = z[i]

            T2 = T[i+1]
            etai = eta_function(T2, pressures[i], **eta_kwargs)
            dTdz_adi = dudx_ambient_function(T2, None, alphai, cpi, gi)

            dudz = [(2*alphai*cpi*dTdz_adi*gi*li**4*rhoi**2 - etai*kci - np.sqrt(etai*(-4*Hi*alphai*cpi*gi*li**4*rhoi**2*zi - 4*alphai*cpi*dTdz_adi*gi*kci*li**4*rhoi**2 + etai*kci**2)))/(2*alphai*cpi*gi*li**4*rhoi**2), (2*alphai*cpi*dTdz_adi*gi*li**4*rhoi**2 - etai*kci + np.sqrt(etai*(-4*Hi*alphai*cpi*gi*li**4*rhoi**2*zi - 4*alphai*cpi*dTdz_adi*gi*kci*li**4*rhoi**2 + etai*kci**2)))/(2*alphai*cpi*gi*li**4*rhoi**2)]

# def initial_ss():
#     def fun(x, ):
#         dTdt = calc_total_heating_rate_numeric(0, u, dx, xprime, l_function, dudx_ambient_function, eta_function, g_function, kc,
#                                     alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l, pressures,
#                                     tspan=None, show_progress=False)
#
#     # find roots
#     from scipy import optimize
#     U_0 = initial_linear(z, u1, u0)
#
#     H = g_function(t=0, x=z, **g_kwargs)
#     dTdz = optimize.root(fun, U_0, args=(l, rho, alpha, cp, k, g, eta, H, dx), jac=None, method='hybr')
#     print('dTdz', dTdz, np.shape(dTdz))
#
#
# def initial_steadystate(z, Tsurf, Tcmb0, l, rho, alpha, cp, kc, g, pressures,
#                         eta_function=None, eta_kwargs=None, g_function=None, g_kwargs={}, dudx_ambient_function=None,
#                         tol=1e-6):
#     N = len(z)
#     dz = z[1] - z[0]
#     zp = (z - z[0]) / (z[-1] - z[0])
#     T = initial_linear(zp, Tsurf, Tcmb0)  # starting guess
#     H = g_function(t=0, x=zp, **g_kwargs)  # constant initial heating
#
#     # thermodynamic parameters const temp
#     cp = [1190] * N  # J/kg/K
#     alpha = [3e-5] * N
#     g = [10] * N
#     kc = [5] * N
#     rho = [4500] * N  # kg/m3
#
#     Fs_old = 0
#     Fs_new = 100000
#
#     T0_old = 100000000000
#     T0_new = T[0]
#
#     iter = 0
#     # iterate on surface flux to get steady state
#     # while abs(Fs_old - Fs_new) > tol:
#     while abs(T0_old - T0_new) > tol:
#
#         Fs_old = Fs_new
#         T0_old = T0_new
#
#         # eta = eta_function(T, pressures, **eta_kwargs)
#         eta = [1e20] * N
#
#         # # from sympy solution - cartesian
#         # dudx = [(2 * alpha * cp * dudx_adiabat * g * l ** 4 * rho ** 2 - eta * kc - np.sqrt(eta * (
#         #             -4 * H * alpha * cp * g * l ** 4 * rho ** 2 * z - 4 * alpha * cp * dudx_adiabat * g * kc * l ** 4 * rho ** 2 + eta * kc ** 2))) / (
#         #              2 * alpha * cp * g * l ** 4 * rho ** 2),
#         #  (2 * alpha * cp * dudx_adiabat * g * l ** 4 * rho ** 2 - eta * kc + np.sqrt(eta * (
#         #              -4 * H * alpha * cp * g * l ** 4 * rho ** 2 * z - 4 * alpha * cp * dudx_adiabat * g * kc * l ** 4 * rho ** 2 + eta * kc ** 2))) / (
#         #              2 * alpha * cp * g * l ** 4 * rho ** 2)]
#         # print('dTdz', dudx)
#         # dudx = dudx[0]  # for now assume 1st solution is the -ve one and the other is +ve
#         # integrate vectorised profile
#         # T[0:N - 2] = T[1:N - 1] - dudx[0:N - 2] * dz
#
#         # fig, ax = plt.subplots(1, 3, figsize=(10, 3))
#         # ax[0].plot(zp, T)
#         # ax[0].set_ylabel('T')
#         # ax[1].plot(zp, dudx)
#         # ax[1].set_ylabel('dT/dz')
#         # ax[2].plot(zp, eta)
#         # ax[2].set_ylabel('eta')
#         # plt.plot(zp, T)
#         # plt.title('iter' + str(iter))
#         # plt.tight_layout()
#         # plt.show()
#
#         # # from sympy solution - cartesian
#         # building profile from top
#         for i in range(N - 2, -1, -1):
#             T2 = T[i + 1]
#             root = -4 * H[i] * alpha[i] * cp[i] * eta[i] * g[i] * l[i] ** 4 * rho[i] ** 2 * z[i] + 4 * T2 * alpha[i] ** 2 * eta[i] * g[i] ** 2 * kc[i] * l[i] ** 4 * rho[i] ** 2 + eta[i] ** 2 * kc[i] ** 2
#             if root < 0:
#                 print('no solution T2', T2, 'i', i, 'root', root)
#                 T2 = T1[1]  # take other root from last time lol
#                 print('new root',-4 * H[i] * alpha[i] * cp[i] * eta[i] * g[i] * l[i] ** 4 * rho[i] ** 2 * z[i] + 4 * T2 * alpha[i] ** 2 * eta[i] * g[i] ** 2 * kc[i] * l[i] ** 4 * rho[i] ** 2 + eta[i] ** 2 * kc[i] ** 2)
#
#
#             T1 = [T2 * alpha[i] * dz * g[i] / cp[i] + T2 + dz * eta[i] * kc[i] / (2 * alpha[i] * cp[i] * g[i] * l[i] ** 4 * rho[i] ** 2) - dz * np.sqrt(
#                 -4 * H[i] * alpha[i] * cp[i] * eta[i] * g[i] * l[i] ** 4 * rho[i] ** 2 * z[i] + 4 * T2 * alpha[i] ** 2 * eta[i] * g[i] ** 2 * kc[i] * l[i] ** 4 * rho[i] ** 2 + eta[i] ** 2 * kc[i] ** 2) / (
#                          2 * alpha[i] * cp[i] * g[i] * l[i] ** 4 * rho[i] ** 2),
#              T2 * alpha[i] * dz * g[i] / cp[i] + T2 + dz * eta[i] * kc[i] / (2 * alpha[i] * cp[i] * g[i] * l[i] ** 4 * rho[i] ** 2) + dz * np.sqrt(
#                  -4 * H[i] * alpha[i] * cp[i] * eta[i] * g[i] * l[i] ** 4 * rho[i] ** 2 * z[i] + 4 * T2 * alpha[i] ** 2 * eta[i] * g[i] ** 2 * kc[i] * l[i] ** 4 * rho[i] ** 2 + eta[i] ** 2 * kc[i] ** 2) / (
#                          2 * alpha[i] * cp[i] * g[i] * l[i] ** 4 * rho[i] ** 2)]
#             print('i', i ,'T', T1, 'K')
#             T[i] = T1[0]
#
#         # # get new surface flux
#         # # eta = eta_function(T=T, P=pressures, **eta_kwargs)  # for initial condition take a nearly constant hot viscosity
#         # dudx_adiabat = dudx_ambient_function(T, z, alpha, cp, g)
#         # dudx = np.gradient(T, dz)
#         # kv = convective_coefficient(alpha, rho, cp, g, l, eta, dudx_adiabat, dudx)
#         # Fs_new = total_heat_transport(kc, kv, dudx, dudx_adiabat)[-2]
#         #
#         # print(iter, 'Fs_new', Fs_new, 'W/m2')
#         T0_new = T[0]
#         print(iter, 'T0_new', T0_new, 'K')
#         iter += 1
#
#
#     # else:
#     #     # find roots
#     #     from scipy import optimize
#     #     U_0 = initial_linear(z, u1, u0)
#     #
#     #     H = g_function(t=0, x=z, **g_kwargs)
#     #
#     #     def fun(x, l, rho, alpha, cp, k, g, eta, H,  dx, Tsurf, Nm):
#     #         T = [Tsurf] * Nm  # upper boundary condition T=0
#     #         for ii in range(Nm - 1, 0, -1):
#     #             T2 = x[ii]
#     #             l2 = l[ii]
#     #             dTdz_ad = dudx_ambient_function(T2, None, alpha, cp, g)
#     #             eta = eta_function(U_0, pressures, **eta_kwargs)
#     #             kv = rho ** 2 * cp * alpha * g * l ** 4 / eta * (x - dTdz_ad)
#     #
#     #         return k * x + kv * (x - dTdz_ad) + (1 / 3) * z * H
#     #
#     #     dTdz = optimize.root(fun, U_0, args=(l, rho, alpha, cp, k, g, eta, H, dx), jac=None, method='hybr')
#     #     print('dTdz', dTdz, np.shape(dTdz))
#     #
#     #     T = np.zeros_like(z)
#     #     T[-1] = Tsurf
#     #     for ii in range(len(T) - 1, 0, -1):
#     #         T[ii - 1] = T[ii] - dTdz[ii] * dz
#
#     return T


def initial_linear(z, u1, u0):
    # initial linear u profile where u0: bottom temperature and u1: top temperature
    # z is nondimensional
    return (u1 - u0) * z + u0


def du0dx(t):
    # bottom boundary condition if on flux ∂u/∂x | (x=0, t) (here x=0, but originally ∂u/∂x | (x=L, t)
    return 0


####################### heat transfer functions ###################

def dudx_ambient(u, x, alpha, cp, gravity):
    """ "ambient" T profile at dimensionless z where alpha, cp, gravity are scalars"""
    return -alpha / cp * gravity * u


# def dudx_ambient_vectors(u, x, alpha, cp, gravity):
#     """ "ambient" T profile at dimensionless x (scalar) where alpha, cp, gravity are vectors"""
#     try:
#         # return entire profile
#         assert np.size(alpha) == np.size(u)
#         return -alpha / cp * gravity * u
#     except AssertionError:
#         # return value at single depth
#         assert np.size(x) == 1
#         n = int(np.round(x * (len(alpha) - 1)))  # get index -- z from 0 to 1 --- confirmed this works for all n
#         return -alpha[n] / cp[n] * gravity[n] * u


def convective_coefficient(alpha, rho, cp, gravity, l, eta, dudx_adiabat, dudx):
    # can't be negative
    return np.maximum(alpha * rho ** 2 * cp * gravity * l ** 4 / (18 * eta) * (dudx_adiabat - dudx), 0)


def total_heat_transport(kc, kv, dudx, dudx_adiabat):
    # get sum of advective and diffusive heat flux terms
    diff_term = -kc * dudx
    adv_term = -kv * (dudx - dudx_adiabat)
    return diff_term + adv_term  # W/m2


# def radiogenic_heating(t, x, H0=3.4e-11, rho=None, t0_buffer_Gyr=0, **kwargs):
#     """Calculate radiogenic heating in W kg^-1 from Korenaga (2006)"""
#     sec2Gyr = 1 / 3.154e7 * 1e-9
#     t_Gyr = t * sec2Gyr + t0_buffer_Gyr
#
#     # order of isotopes: 238U, 235U, 232Th, 40K
#     c_n = np.array([0.9927, 0.0072, 4.0, 1.6256])  # relative concentrations of radiogenic isotopes
#     p_n = np.array([9.37e-5, 5.69e-4, 2.69e-5, 2.79e-5])  # heat generation rates (W/kg)
#     lambda_n = np.array([0.155, 0.985, 0.0495, 0.555])  # half lives (1/Gyr)
#     h_n = (c_n * p_n) / np.sum(c_n * p_n)  # heat produced per kg of isotope normalized to total U
#     # print('H0', H0, 'h_n', h_n, 'lambda_n', lambda_n, 't_Gyr', t_Gyr, 'exp', sum(h_n * np.exp(lambda_n * t_Gyr)) )
#     h = H0 * sum(h_n * np.exp(-lambda_n * t_Gyr))
#     # print('h', h, 'h*rho', h*rho, 'rho', rho)
#     return h * rho


# def internal_heating_decaying(t, x, age_Gyr=None, x_Eu=1, rho=None, t0_buffer_Gyr=0):
#     """ radiogenic heating in W/kg after Table 1 and eqn 1 in O'Neill+ 2020 (SSR)
#     x_Eu: concentration of r-process elements wrt solar (i.e. Eu, U, Th)"""
#     # order of isotopes (IMPORTANT) is [40K, 238U, 235U, 232Th]
#     tau_i = np.array([1250, 4468, 703.8, 14050])  # half life in Myr
#     h_i = np.array([28.761e-6, 94.946e-6, 568.402e-6, 26.368e-6])  # heat production in W/kg
#     c_i = np.array([30.4e-9, 22.7e-9, 0.16e-9, 85e-9])  # BSE concentration in kg/kg
#
#     sec2Gyr = 1 / 3.154e7 * 1e-9
#
#     # convert times to Myr to be consistent with units of tau
#     # add buffer to start simulation from x Gyr (solver has problems if t0>0 ?)
#     t_Myr = (t * sec2Gyr + t0_buffer_Gyr) * 1e3
#     h_K = np.array(c_i[0] * h_i[0] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[0]))
#     try:
#         h_UTh = np.array(sum(c_i[1:] * h_i[1:] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[1:])))
#     except ValueError:
#         t_Myr = np.vstack((t_Myr, t_Myr, t_Myr))
#         c_i = c_i.reshape((4, 1))
#         h_i = h_i.reshape((4, 1))
#         tau_i = tau_i.reshape((4, 1))
#         h_UTh = np.array(np.sum(c_i[1:] * h_i[1:] * np.exp((age_Gyr * 1e3 - t_Myr) * np.log(2) / tau_i[1:]), axis=0))
#
#     h_perkg = (h_K + x_Eu * h_UTh)
#     # print(t * sec2Gyr * 1e3, 'Myr', 'h', h_perkg, 'W/kg')
#     return h_perkg * rho


def rad_heating_forward(t, x, rho, rad_factor=1, t_buffer_Gyr=0, **kwargs):
    # O'Neill+ SSR 2020
    # 40K, 238U, 235U, 232Th
    t_Myr = t / years2sec * 1e-6  # + t_buffer_Gyr * 1e3

    c0 = np.array([30.4e-9, 22.7e-9, 0.16e-9, 85e-9])  # present day BSE concentration
    c0[1:] = c0[1:] * rad_factor  # scale refractories
    hn = np.array([28.761e-6, 94.946e-6, 568.402e-6, 26.368e-6])  # heating rate per kg isotope
    tau = np.array([1250, 4468, 703.8, 14040])  # half life in Myr
    H0 = c0 * hn * np.exp(4500 * np.log(2) / tau)  # initial heating based on (scaled) BSE concentrations

    H = np.sum(H0 * np.exp((-t_Myr) * np.log(2) / tau))
    return H * rho  # W m-3


def internal_heating_constant(t, x, H0=1e-12, **kwargs):
    return H0  # W m-3


######################## solve ode in t ###########################


def solve_pde(t0, tf, U_0, heating_rate_function, ivp_kwargs, max_step=1e6 * years2sec,
              verbose=False, show_progress=False, save_progress=False):
    """
    :param save_progress:
    :type save_progress:
    :param writefile:
    :type writefile:
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
    # from inspect import signature
    # from tqdm import tqdm

    # if verbose:
    #     # check function signature of heating rate function matches ivp_args?
    #     sig = signature(heating_rate_function)
    #     print('\nrequired signature for ivp_args:', str(sig))  # skipping t, u
    #     print('ivp_args', ivp_args)
    #     # print('signature of heating_rate_function', list(sig.parameters.keys())[2:])  # skipping t, u

    # for use with solve_ivp, need extra args as non-keyword args
    ivp_args = tuple([ivp_kwargs[k] for k in ivp_kwargs.keys()])

    tspan = (t0, tf)
    ivp_args = ivp_args + (tspan, show_progress, save_progress)

    if verbose:
        print('Solving IVP from', tspan[0] /years2sec * 1e-9, 'to', tspan[1] / years2sec * 1e-9, 'Gyr')

    start = time.time()
    soln = solve_ivp(heating_rate_function,
                     t_span=tspan, y0=U_0,
                     method='BDF',  # conserves energy
                     # method='LSODA',
                     t_eval=None,
                     vectorized=False,
                     args=ivp_args,
                     max_step=max_step,
                     jac=None,  # recommended for BDF, todo
                     # **kwargs
                     )
    end = time.time()

    print('\n')
    if verbose:
        print(len(soln.t), 'timesteps in', end - start, 'seconds', '(' + heating_rate_function.__name__ + ')')
        print('    u', np.shape(soln.y), ', t', np.shape(soln.t))
    return soln


################### different methods to calculate total heating rate on RHS ###########################


def calc_total_heating_rate_numeric(t, u, dx, xprime, l_function, dudx_ambient_function, eta_function, g_function, kc,
                                    alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l, pressures,
                                    tspan, show_progress, save_progress):
    """ function to calculate dT/dt for each z (evaluated at array of temperatures u)
    this can be sped up by ~2 if mixing length is time-independent"""
    percent_complete = (((t - tspan[0]) / (tspan[1] - tspan[0])) * 100)

    if show_progress:
        print("\rModel time: " + str(format(t / years2sec, ".0f")) + " yr, " + str(
            format(percent_complete, ".4f")) + "% complete",
              end='', flush=True)

    if save_progress:
        with open(save_progress, "wb") as pfile:
            pkl.dump((t, u), pfile)

    # update viscosity
    eta = eta_function(u, pressures, **eta_kwargs)

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
    q = total_heat_transport(kc, kv, dudx, dudx_adiabat)
    divq = np.gradient(q, dx)

    lhs = -divq + source_term

    # redo boundary conditions (np.gradient might do this differently)
    lhs[-1] = 0  # value of du/dt at x=L - i.e. constant temperature so no dT/dt

    # bottom bdy condition using constant flux - can assume diffusion only
    try:
        lhs[0] = (kc / dx ** 2) * (2 * u[1] + 2 * dx * du0dx(t) - 2 * u[0]) + source_term
    except ValueError:
        # source term and/or kc is a vector - todo make it always a vector?
        lhs[0] = (kc[0] / dx ** 2) * (2 * u[1] + 2 * dx * du0dx(t) - 2 * u[0]) + source_term[0]

    # # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
    # lhs[0] = 0  # value of du/dt at x=L - i.e. constant temperature so no dT/dt

    dudt = lhs / (rho * cp)

    # print('dudt', dudt)
    # print('eta', eta)
    # print('H', source_term)
    # print('diff term', diff_term)
    # print('adv term', adv_term)

    return dudt


#
# def calc_total_heating_rate_analytic_isoviscous(t, u, dx, dudx_ambient_function, g_function, l, dldx, kc,
#                                                 alpha, rho, cp, gravity, eta, g_kwargs):
#     """ # function to calculate dT/dt for each z (evaluated at array of temperatures u).
#      like the above but as analytic as possible
#      for isoviscous case this is  ~0.1 sec faster
#      """
#     N = len(u)
#     rhs = np.zeros(N)
#     rhs[-1] = 0  # surface boundary condution, dT/dt = 0 because constant T
#
#     source_term = g_function(t, None, **g_kwargs)  # assuming no depth-dependence
#
#     # isoviscous case can be done analytically
#     A = alpha * rho ** 2 * cp * gravity * l ** 4 / (18 * eta)
#     dAdx = 4 * alpha * rho ** 2 * cp * gravity * l ** 3 / (18 * eta) * dldx
#
#     # building profile from top
#     for i in range(N - 2, 0, -1):
#
#         # discretize derivatives - central differences
#         d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)  # second order finite difference
#         dudx = (u[i + 1] - u[i - 1]) / (2 * dx)  # central difference
#         dudx_adiabat = dudx_ambient_function(u[i], None, alpha, cp, gravity)
#         d2udx2_adiabat = -alpha / cp * gravity * dudx  # temp - assuming virtually constant alpha, cp, g
#
#         # check for convection
#         if abs(dudx_adiabat) > abs(dudx):  # -ve kv, no convection
#             rhs[i] = kc * d2udx2 + source_term
#         #             print('kv = 0 at z =', i)
#         else:
#             rhs[i] = kc * d2udx2 - dAdx[i] * (dudx ** 2 - 2 * dudx * dudx_adiabat + dudx_adiabat ** 2) \
#                      - 2 * A[i] * (
#                              dudx * d2udx2 - d2udx2 * dudx_adiabat - dudx * d2udx2_adiabat + dudx_adiabat * d2udx2_adiabat) \
#                      + source_term
#
#     #     # bottom bdy condition using constant flux - can assume diffusion only
#     #     rhs[0] = (kc/dx**2)*(2*u[1] + 2*dx*du0dx(t) - 2*u[0]) + source_term
#
#     # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
#     rhs[0] = 0
#
#     return rhs / (rho * cp)
#
#
# # function to calculate dT/dt for each z (evaluated at array of temperatures u)
# # like the above but as analytic as possible
# def calc_total_heating_rate_analytic(t, u, dx, xprime, l_function, dudx_ambient_function, eta_function, g_function, kc,
#                                      alpha, rho, cp,
#                                      gravity, L, l_kwargs, eta_kwargs, g_kwargs):
#     N = len(u)
#     rhs = np.zeros(N)
#     rhs[-1] = 0  # surface boundary condution, dT/dt = 0 because constant T
#
#     source_term = g_function(t, xprime, **g_kwargs)
#
#     # update eta(T)
#     eta = eta_function(u, xprime, **eta_kwargs)
#
#     # isoviscous case can be done analytically
#     lp, dldx = l_function(xprime, **l_kwargs)
#     l = lp * L  # dimensionalise
#     f = alpha * rho ** 2 * cp * gravity / (18 * eta)
#     df = np.gradient(f, dx)  # a bit too complicated to do analytically
#     g = l ** 4
#     dg = 4 * l ** 3 * dldx
#
#     A = f * g
#
#     try:
#         dAdx = (f * dg) + (df * g)
#     except ValueError:
#         assert len(df) == 0  # f is a scalar, df is here an empty list (actually =0)
#         dAdx = f * dg
#
#     # fig, ax = plt.subplots(1, 2)
#     # ax[0].plot(zp, A)
#     # ax[1].plot(zp, dAdx)
#     # ax[0].set_ylabel('A')
#     # ax[1].set_ylabel('dAdx')
#     # ax[0].legend(['t={:.3f} Myr'.format(t / years2sec * 1e-6)])
#     # plt.show()
#
#     # building profile from top
#     for i in range(N - 2, 0, -1):
#
#         # discretize derivatives - central differences
#         d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2)  # second order finite difference
#         dudx = (u[i + 1] - u[i - 1]) / (2 * dx)  # central difference
#         dudx_adiabat = dudx_ambient_function(u[i], xprime, alpha, cp, gravity)
#         d2udx2_adiabat = -alpha / cp * gravity * dudx  # temp - assuming virtually constant alpha, cp, g
#
#         # check for convection
#         if abs(dudx_adiabat) > abs(dudx):  # -ve kv, no convection
#             rhs[i] = kc * d2udx2 + source_term
#         #             print('kv = 0 at z =', i)
#         else:
#             rhs[i] = kc * d2udx2 - dAdx[i] * (dudx ** 2 - 2 * dudx * dudx_adiabat + dudx_adiabat ** 2) \
#                      - 2 * A[i] * (
#                              dudx * d2udx2 - d2udx2 * dudx_adiabat - dudx * d2udx2_adiabat + dudx_adiabat * d2udx2_adiabat) \
#                      + source_term
#
#         # bottom bdy condition using constant flux - can assume diffusion only
#         rhs[0] = (kc/dx**2)*(2*u[1] + 2*dx*du0dx(t) - 2*u[0]) + source_term
#
#     # # alternatively, for a constant T bottom boundary condition, fix at Tcmb0 with du/dt=0:
#     # rhs[0] = 0
#
#     return rhs / (rho * cp)


############################### tests #############################################

def test_isoviscous(N=500, Nt_min=0, writefile=None, verbose=True, plot=True):
    """ test generic case """
    from MLTMantle import get_mixing_length_and_gradient_smooth, Arrhenius_viscosity_law
    import matplotlib.pyplot as plt
    from PlanetInterior import pt_profile
    from MLTMantle import save_h5py_solution

    def internal_heating_constant(t, x, **kwargs):
        return 1e-12  # W kg-1, for testing

    def viscosity_constant(u, x, **kwargs):
        return 10 ** 21

    # set up grid
    Rc, Rp = 3475e3, 6370e3
    L = Rp - Rc  # length scale
    D = 1  # dimensionless length scale
    zp = np.linspace(0, 1, N)  # dimensionless height

    # increments and domain
    dx = (zp[1] - zp[0]) * L
    t0, tf = 0, 5e9 * years2sec

    try:
        max_step = (tf - t0) / Nt_min
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

    # constant pressure structure - evaluate at some Tp but should be roughly independent of Tp
    pressures = pt_profile(N, radius=zp * (Rp - Rc) + Rc, density=[rho] * N, gravity=[gravity] * N, alpha=[alpha] * N,
                           cp=[cp] * N, psurf=1, Tp=1700)  # Pa

    U_0 = initial_linear(zp, Tsurf, Tcmb0)  # initial temperature

    ivp_args = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient, viscosity_constant,
                 internal_heating_constant, kc, alpha, rho, cp, gravity, L, l_kwargs, eta_kwargs, g_kwargs, l,
                 pressures)

    ivp_kwargs = {'dx': dx, 'zp': zp, 'get_mixing_length_and_gradient_smooth': get_mixing_length_and_gradient_smooth,
                  'dudx_ambient':dudx_ambient, 'viscosity_constant': viscosity_constant,
                'internal_heating_constant': internal_heating_constant, 'kc': kc, 'alpha': alpha,
                  'rho': rho, 'cp': cp, 'gravity': gravity, 'L': L,
                'l_kwargs': l_kwargs, 'eta_kwargs': eta_kwargs, 'g_kwargs': g_kwargs, 'l': l,
                'pressures': pressures}  # needs to match signature to heating_rate_function}

    soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args, verbose=verbose, show_progress=True,
                      max_step=max_step)

    if writefile:
        save_h5py_solution(writefile, soln, ivp_kwargs)


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


def test_arrhenius_radheating(N=1000, Nt_min=1000, t_buffer_Myr=0, age_Gyr=4.5, verbose=True, writefile=None, plot=True,
                              figpath=None):
    """ test generic case """
    from MLTMantle import get_mixing_length_and_gradient_smooth, exponential_viscosity_law, Arrhenius_viscosity_law
    from MLTMantleCalibrated import get_mixing_length_calibration
    import matplotlib.pyplot as plt
    from PlanetInterior import pt_profile
    from MLTMantle import save_h5py_solution

    # set up grid/domain
    Rc, Rp = 3475e3, 6370e3
    L = Rp - Rc  # length scale
    D = 1  # dimensionless length scale
    zp = np.linspace(0, 1, N)  # dimensionless height
    dx = (zp[1] - zp[0]) * L
    t0, tf = t_buffer_Myr * 1e6 * years2sec, age_Gyr * 1e9 * years2sec  # seconds

    try:
        max_step = (tf - t0) / Nt_min
        if verbose:
            print('max step:', max_step / years2sec, 'years')
    except ZeroDivisionError:
        max_step = np.inf
        if verbose:
            print('max step: inf')

    # dimensionless convective parameters
    RaH = 1e7
    dEta = 1e5

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

    # constant pressure structure - evaluate at some Tp but should be roughly independent of Tp
    pressures = pt_profile(N, radius=zp * (Rp - Rc) + Rc, density=[rho] * N, gravity=[gravity] * N, alpha=[alpha] * N,
                           cp=[cp] * N, psurf=1, Tp=1700)  # Pa

    # boundary conditions
    Tsurf = 300
    Tcmb0 = 2850
    U_0 = initial_linear(zp, Tsurf, Tcmb0)  # initial temperature

    l_kwargs = {'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt}
    # g_kwargs_constant = {'rho': rho, 'H': H0}  # not relevant here but args passed to g_function
    g_kwargs_decay = {'rho': rho, 't_buffer_Gyr': t_buffer_Myr * 1e-3}
    # eta_kwargs_Arr = {'eta_ref': 1e21, 'T_ref': 1600, 'Ea': 300e3}  # no pressure-dependence, with ref. viscosity
    eta_kwargs_Arr = {}  # Tachninami

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
    # eta_b = Arrhenius_viscosity_law(Tcmb0, None, **eta_kwargs_Arr)
    # print('eta_b', eta_b)

    ivp_args = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient, Arrhenius_viscosity_law,
                rad_heating_forward, kc, alpha, rho, cp, gravity, L,
                l_kwargs, eta_kwargs_Arr, g_kwargs_decay, l,
                pressures)  # needs to match signature to heating_rate_function

    ivp_kwargs = {'dx': dx, 'zp': zp, 'get_mixing_length_and_gradient_smooth': get_mixing_length_and_gradient_smooth,
                  'dudx_ambient':dudx_ambient, 'eta_function': Arrhenius_viscosity_law,
                'g_function': rad_heating_forward, 'kc': kc, 'alpha': alpha,
                  'rho': rho, 'cp': cp, 'gravity': gravity, 'L': L,
                'l_kwargs': l_kwargs, 'eta_kwargs': eta_kwargs_Arr, 'g_kwargs': g_kwargs_decay, 'l': l,
                'pressures': pressures}  # needs to match signature to heating_rate_function}

    soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_args,
                     verbose=True, show_progress=True, max_step=max_step)

    if writefile:
        save_h5py_solution(writefile, soln, ivp_kwargs)


    if plot:
        # plot
        fig, ax = plt.subplots(1, 3)

        n = -1
        ax[0].plot(zp, soln.y[:, n], label='Arrhenius')
        # plt.plot(zp, soln2.y[:, n], label='exponential')
        ax[0].set_xlabel('z/L')
        ax[0].set_ylabel('T (K)')
        ax[0].set_title('t={:.3f} Myr'.format(soln.t[n] / years2sec * 1e-6))
        ax[0].legend()

        # i = N - 1
        # plt.figure()
        # plt.plot(soln.t / years2sec * 1e-6, soln.y[i, :])
        # plt.xlabel('time (Myr)')
        # plt.ylabel('Surface temperature (K)')

        n = 0  # initial
        ax[1].plot(zp, np.log10(Arrhenius_viscosity_law(soln.y[:, n], zp, **eta_kwargs_Arr)), label='Arrhenius')
        # ax[1].plot(zp, np.log10(
        #     exponential_viscosity_law(soln2.y[:, n], zp, **eta_kwargs2)), label='exponential')
        ax[1].set_xlabel('z/L')
        ax[1].set_ylabel(r'log$\eta$')
        ax[1].set_title('t={:.3f} Myr'.format(soln.t[n] / years2sec * 1e-6))

        # internal heating
        ax[2].plot(soln.t / (years2sec * 1e9), [rad_heating_forward(tt, x=None, **g_kwargs_decay) for tt in soln.t],
                   label='Radiogenic heating')
        ax[2].set_xlabel('t (Gyr)')
        ax[2].set_ylabel('H (W/m3)')
        ax[2].legend()

        plt.tight_layout()
        if figpath is not None:
            fig.savefig(figpath, bbox_inches='tight')
        else:
            plt.show()


def test_pdependence(N=1000, Nt_min=1000, t_buffer_Myr=0, age_Gyr=4.5, verbose=True, writefile=None, plot=True,
                     figpath=None, save_progress=None, Mantle=None, cmap='magma'):
    """ test generic case """
    from MLTMantle import (get_mixing_length_and_gradient_smooth, Arrhenius_viscosity_law_pressure)
    # from MLTMantleCalibrated import get_mixing_length_calibration
    from PlanetInterior import pt_profile
    from MLTMantle import save_h5py_solution

    # dimensionless convective parameters
    # RaH = 1e7
    # dEta = 1e5
    # mixing length calibration (stagnant lid mixed heated)
    # alpha_mlt = 0.2895
    # beta_mlt = 0.6794
    # alpha_mlt, beta_mlt = get_mixing_length_calibration(RaH, dEta)
    # if verbose:
    #     print('alpha_mlt', alpha_mlt, 'beta_mlt', beta_mlt)

    if Mantle is None:
        # set up grid/domain
        Rc, Rp = 3475e3, 6370e3
        zp = np.linspace(0, 1, N)  # dimensionless height

        # thermodynamic parameters
        cp = 1190  # J/kg/K
        alpha = 3e-5
        gravity = 10
        kc = 5
        rho = 4500  # kg/m3
        kappa = kc / (rho * cp)

        # boundary conditions
        Tsurf = 300
        Tcmb0 = 2500  # only used for initial condition, bc is constant flux

        # constant pressure structure - evaluate at some Tp but should be roughly independent of Tp
        pressures, Tp = pt_profile(N, radius=zp * (Rp - Rc) + Rc, density=[rho] * N, gravity=[gravity] * N,
                                   alpha=[alpha] * N, cp=[cp] * N, psurf=1, Tp=1700)  # Pa

    else:
        # set up grid/domain
        N = Mantle.Nm
        Rp, Rc = Mantle.r[-1], Mantle.r[0]
        zp = Mantle.zp

        # thermodynamic paramters
        cp = Mantle.cp_m
        alpha = Mantle.alpha_m
        gravity = Mantle.g_m
        kc = Mantle.k_m
        rho = Mantle.rho_m
        kappa = Mantle.kappa_m

        # boundary conditions
        Tsurf = Mantle.Tsurf
        Tcmb0 = Mantle.Tcmb0  # only used for initial condition, bc is constant flux

        # pressure profile
        pressures = Mantle.P
        Tp = Mantle.T_adiabat

    L = Rp - Rc  # length scale
    D = 1  # dimensionless length scale
    dx = (zp[1] - zp[0]) * L
    t0, tf = t_buffer_Myr * 1e6 * years2sec, age_Gyr * 1e9 * years2sec  # seconds

    try:
        max_step = (tf - t0) / Nt_min
        if verbose:
            print('max step:', max_step / years2sec, 'years')
    except ZeroDivisionError:
        max_step = np.inf
        if verbose:
            print('max step: inf')

    # MLT constants
    alpha_mlt, beta_mlt = 0.82, 1  # Tachinami 2011
    lp, dldx = get_mixing_length_and_gradient_smooth(zp, alpha_mlt, beta_mlt)
    l = lp * L

    # initial T profile - from file
    # U_0 = initial_linear(zp, Tsurf, Tcmb0)  # initial temperature
    U_0 = initial_file("Tachinami.h5py", outputpath="output/tests/")

    l_kwargs = {'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt}
    # g_kwargs_constant = {'rho': rho, 'H': H0}  # not relevant here but args passed to g_function
    g_kwargs_decay = {'rho': rho, 't_buffer_Myr': t_buffer_Myr}
    eta_kwargs = {}  # Tackley - kwargs hardcoded into function for the time being

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
    # eta_b = Arrhenius_viscosity_law(Tcmb0, None, **eta_kwargs_Arr)
    # print('eta_b', eta_b)

    ivp_args = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient, Arrhenius_viscosity_law_pressure,
                rad_heating_forward, kc, alpha, rho, cp, gravity, L,
                l_kwargs, eta_kwargs, g_kwargs_decay, l,
                pressures)  # needs to match signature to heating_rate_function

    ivp_kwargs = {'dx': dx, 'zp': zp, 'get_mixing_length_and_gradient_smooth': get_mixing_length_and_gradient_smooth,
                  'dudx_ambient':dudx_ambient, 'eta_function': Arrhenius_viscosity_law_pressure,
                'g_function': rad_heating_forward, 'kc': kc, 'alpha': alpha,
                  'rho': rho, 'cp': cp, 'gravity': gravity, 'L': L,
                'l_kwargs': l_kwargs, 'eta_kwargs': eta_kwargs, 'g_kwargs': g_kwargs_decay, 'l': l,
                'pressures': pressures}  # needs to match signature to heating_rate_function}

    soln = solve_pde(t0, tf, U_0, calc_total_heating_rate_numeric, ivp_kwargs,
                     verbose=True, show_progress=True, max_step=max_step, save_progress=save_progress)

    if writefile:
        save_h5py_solution(writefile, soln, ivp_kwargs)

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cmx

        # colourise
        cm = plt.get_cmap(cmap)
        cNorm = mcolors.Normalize(vmin=soln.t[0], vmax=soln.t[-1])
        scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        c = scalarmap.to_rgba(soln.t)

        # temperature evolution
        fig = plt.figure()
        for n in np.arange(len(soln.t))[::100]:
            plt.plot(zp, soln.y[:, int(n)], c=c[n])
        plt.xlabel('z/L')
        plt.ylabel('T (K)')
        plt.colorbar(
            plt.gca().scatter(soln.t / years2sec * 1e-6, soln.t / years2sec * 1e-6, c=soln.t / years2sec * 1e-6,
                              cmap='magma', s=0), label='time (Myr)')

        plt.tight_layout()
        if figpath is not None:
            fig.savefig(figpath, bbox_inches='tight')
        else:
            plt.show()


def solve_hot_initial(N=1000, Nt_min=1000, t_buffer_Gyr=4.5, verbose=True, writefile=None, plot=True,
                     figpath=None, Mantle=None, cmap='magma'):
    """ test generic case """
    from MLTMantle import (get_mixing_length_and_gradient_smooth, Arrhenius_viscosity_law_pressure)
    from PlanetInterior import pt_profile

    # dimensionless convective parameters
    # RaH = 1e7
    # dEta = 1e5
    # mixing length calibration (stagnant lid mixed heated)
    # alpha_mlt = 0.2895
    # beta_mlt = 0.6794
    # alpha_mlt, beta_mlt = get_mixing_length_calibration(RaH, dEta)
    # if verbose:
    #     print('alpha_mlt', alpha_mlt, 'beta_mlt', beta_mlt)

    if Mantle is None:
        # set up grid/domain
        Rc, Rp = 3475e3, 6370e3
        zp = np.linspace(0, 1, N)  # dimensionless height

        # thermodynamic parameters
        cp = 1190  # J/kg/K
        alpha = 3e-5
        gravity = 10
        kc = 5
        rho = 4500  # kg/m3
        kappa = kc / (rho * cp)

        # boundary conditions
        Tsurf = 300
        Tcmb0 = 3000  # only used for initial condition, bc is constant flux

        # constant pressure structure - evaluate at some Tp but should be roughly independent of Tp
        pressures, Tp = pt_profile(N, radius=zp * (Rp - Rc) + Rc, density=[rho] * N, gravity=[gravity] * N,
                                   alpha=[alpha] * N, cp=[cp] * N, psurf=1, Tp=1700)  # Pa

    else:
        # set up grid/domain
        N = Mantle.Nm
        Rp, Rc = Mantle.r[-1], Mantle.r[0]
        zp = Mantle.zp

        # thermodynamic paramters
        cp = Mantle.cp_m
        alpha = Mantle.alpha_m
        gravity = Mantle.g_m
        kc = Mantle.k_m
        rho = Mantle.rho_m
        kappa = Mantle.kappa_m

        # boundary conditions
        Tsurf = Mantle.Tsurf
        Tcmb0 = Mantle.Tcmb0  # only used for initial condition, bc is constant flux

        # pressure profile
        pressures = Mantle.P
        Tp = Mantle.T_adiabat

    L = Rp - Rc  # length scale
    D = 1  # dimensionless length scale
    dx = (zp[1] - zp[0]) * L
    t0, tf = 0, t_buffer_Gyr * 1e9 * years2sec  # seconds

    try:
        max_step = (tf - t0) / Nt_min
        if verbose:
            print('max step:', max_step / years2sec, 'years')
    except ZeroDivisionError:
        max_step = np.inf
        if verbose:
            print('max step: inf')

    # MLT constants
    alpha_mlt, beta_mlt = 0.82, 1  # Tachinami 2011
    lp, dldx = get_mixing_length_and_gradient_smooth(zp, alpha_mlt, beta_mlt)
    l = lp * L

    U_0 = initial_linear(zp, Tsurf, Tcmb0)  # initial temperature
    # U_0 = initial_file("Tachinami.h5py", outputpath="output/tests/")

    l_kwargs = {'alpha_mlt': alpha_mlt, 'beta_mlt': beta_mlt}
    # g_kwargs_constant = {'rho': rho, 'H': H0}  # not relevant here but args passed to g_function
    g_kwargs_decay = {'rho': rho}
    # eta_kwargs_Arr = {'eta_ref': 1e21, 'T_ref': 1600, 'Ea': 300e3}  # no pressure-dependence, with ref. viscosity
    eta_kwargs = {}  # Tackley - kwargs hardcoded into function for the time being

    H0 = rad_heating_forward(t=0, x=zp, **g_kwargs_decay)  # get initial heating rate
    g_kwargs_constant = {'H0': H0}
    print('H0 =', H0, 'W/kg')

    ivp_args = (dx, zp, get_mixing_length_and_gradient_smooth, dudx_ambient, Arrhenius_viscosity_law_pressure,
                internal_heating_constant, kc, alpha, rho, cp, gravity, L,
                l_kwargs, eta_kwargs, g_kwargs_constant, l,
                pressures)  # needs to match signature to heating_rate_function
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
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as cmx

        # colourise
        cm = plt.get_cmap(cmap)
        cNorm = mcolors.Normalize(vmin=soln.t[0], vmax=soln.t[-1])
        scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        c = scalarmap.to_rgba(soln.t)

        # temperature evolution
        fig = plt.figure()
        for n in np.arange(len(soln.t))[::100]:
            plt.plot(zp, soln.y[:, int(n)], c=c[n])
        plt.xlabel('z/L')
        plt.ylabel('T (K)')
        plt.colorbar(
            plt.gca().scatter(soln.t / years2sec * 1e-6, soln.t / years2sec * 1e-6, c=soln.t / years2sec * 1e-6,
                              cmap='magma', s=0), label='time (Myr)')

        plt.tight_layout()
        if figpath is not None:
            fig.savefig(figpath, bbox_inches='tight')
        else:
            plt.show()


def get_max_step(t0, tf, max_step, Nt_min, verbose=False):
    # default is to use max step if given, else calculate from min number of time steps
    if max_step is None:
        try:
            max_step = (tf - t0) / Nt_min
            if verbose:
                print('max step:', max_step / years2sec, 'years')
        except ZeroDivisionError:
            max_step = np.inf
            if verbose:
                print('max step: inf')
    return max_step
