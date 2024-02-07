"""
Calculate planet interior structure and density-radius profile
(ideally a few options here - ideally use BurnMan)
"""

import numpy as np
import math
import parameters as p


def g_profile(n, radius, density, old_g_quadrature=False):
    # gravity has boundary condition 0 in center (and not at surface)
    # analogously, a surface gravity can be determind from M and
    # guessed R, and gravity can be interpolated from surface downwards
    # Problem with that approach: neg. gravity values in center possible
    gravity = np.zeros(n)
    for i in range(1, n):
        dr = radius[i] - radius[i - 1]
        gravity[i] = (radius[i - 1] ** 2 * gravity[i - 1] + 4 * np.pi * G / 3 * density[i] * (
                radius[i] ** 3 - radius[i - 1] ** 3)) / radius[i] ** 2
    return gravity


def pt_profile(n, radius, density, gravity, alpha, cp, psurf, tsurf, i_cmb=None, deltaT_cmb=0, extrapolate_nan=False):
    """ input psurf in bar, pressure and temperature are interpolated from surface downwards """
    pressure = np.zeros(n)
    temperature = np.zeros(n)
    pressure[-1] = psurf * 1e5  # surface pressure in Pa
    temperature[-1] = tsurf  # potential surface temperature, not real surface temperature
    for i in range(2, n + 1):  # M: 1:n-1; n-i from n-1...1; P: from n-2...0 -> i from 2...n
        dr = radius[n - i + 1] - radius[n - i]
        pressure[n - i] = pressure[n - i + 1] + dr * gravity[n - i] * density[n - i]
        lapse_rate = alpha[n - i] / cp[n - i] * gravity[n - i]
        temperature[n - i] = temperature[n - i + 1] + dr * lapse_rate * temperature[n - i + 1]
        if n - i == i_cmb:
            # add temperature jump across cmb (discontinuity)
            temperature[n - i] = temperature[n - i] + deltaT_cmb
    return pressure, temperature  # Pa, K


def radius_from_bulkdensity(CMF, Mp, rho_c, rho_m):  # from bulk i.e. average values - just for initial guess
    if CMF > 0:
        Rc = (3 * CMF * Mp / (4 * math.pi * rho_c)) ** (1 / 3)
    else:
        Rc = 0
    Rp = (Rc ** 3 + 3 * (1 - CMF) * Mp / (4 * math.pi * rho_m)) ** (1 / 3)
    return Rc, Rp


def iterate_structure(M=p.M_E, CMF=0.325, Psurf=1000, Tp=1600, n='auto', maxIter=100, tol=0.0001,
                      deltaT_cmb=0, rho_m0=None, parameterise_lm=True,
                      **kwargs):
    """ tweaked from Noack+
        Tsurf is potential surface temperature in K (was 1600), Psurf is surface pressure in bar
        n is radial resolution from center of planet to surface, must be <~ 2000 else perple_x error
        tol is fractional convergence criterion for iteration solver
        rho_m0 is initial guess for mantle density (optional)
        parameterise_lm extrapolates a constant composition deeper than 200 GPa"""
    import eos
    import math

    Mc = M * CMF
    x_Fe = CMF  # this is just for starting guess on Rp from parameterisation

    if n == 'auto':
        if M / p.M_E <= 1:
            n = 1200  # 300  # n = 200 saves about 7 sec per run, misses a little Aki phase
        elif M / p.M_E <= 2:
            n = 1200  # 500
        elif M / p.M_E <= 2.5:
            n = 1200
        else:
            n = 1600

    # Initialization - guesses
    cp_c = 800  # guess for core heat capacity in J/kg K
    cp_m = 1300  # guess for mantle heat capacity in J/kg K
    alpha_c = 0.00001  # guess for core thermal expansion ceoff. in 1/K
    alpha_m = 0.000025  # guess for mantle thermal expansion ceoff. in 1/K
    Rp = 1e3 * (7030 - 1840 * x_Fe) * (
            M / p.M_E) ** 0.282  # initial guesss, Noack & Lasbleis 2020 (5) ignoring mantle Fe
    if CMF > 0:
        Rc = 1e3 * 4850 * x_Fe ** 0.328 * (M / p.M_E) ** 0.266  # initial guess, hot case, ibid. (9)
        rho_c_av = x_Fe * M / (4 / 3 * np.pi * Rc ** 3)
    else:
        Rc = 0
        rho_c_av = 0
    if rho_m0 is None:
        rho_m_av = (1 - x_Fe) * M / (4 / 3 * np.pi * (Rp ** 3 - Rc ** 3))  # Noack & Lasbleis parameterisation
    else:
        rho_m_av = rho_m0  # use initial guess as given
        Rc, Rp = radius_from_bulkdensity(x_Fe, M, rho_c_av, rho_m_av)  # get consistent radius

    # Arrays
    radius = np.zeros(n)  # corresponds to height at top of layer
    density = np.zeros(n)
    alpha = np.zeros(n)
    cp = np.zeros(n)
    mass = np.zeros(n)  # cumulative mass, not differential

    # Initialization of arrays: surface values
    radius[-1] = Rp  # radius of top of layer, from centre
    density[-1] = rho_m_av
    alpha[-1] = alpha_m
    cp[-1] = cp_m
    mass[-1] = M
    if x_Fe == 1:  # pure iron shell
        density[-1] = rho_c_av
        alpha[-1] = alpha_c
        cp[-1] = cp_c
        Rc = Rp

    # Initialization of arrays: interpolation over depth from top
    for i in range(2, n + 1):  # goes to i = n-1
        mass[n - i] = mass[n - i + 1] - M / n  # such that it never goes to 0 (would cause numerical errors)
    if CMF > 0:
        i_cmb = np.argmax(mass > Mc)  # index of cmb in profiles
    else:
        i_cmb = 0

    density[i_cmb + 1:] = rho_m_av
    alpha[i_cmb + 1:] = alpha_m
    cp[i_cmb + 1:] = cp_m
    density[:i_cmb + 1] = rho_c_av
    alpha[:i_cmb + 1] = alpha_c
    cp[:i_cmb + 1] = cp_c

    # get radius from volumes
    for i in range(2, n + 1):  # goes to i = n-1
        dmass = mass[n - i + 1] - mass[n - i]
        radius[n - i] = np.cbrt(-(dmass / density[n - i] / (4 * np.pi / 3)) + radius[n - i + 1] ** 3)

    gravity = eos.g_profile(n, radius, density)
    pressure, temperature = eos.pt_profile(n, radius, density, gravity, alpha, cp, Psurf, Tp, i_cmb, deltaT_cmb)
    p_cmb = pressure[i_cmb]  # the pressure (at cell top edge) of the cell that Rc passes through
    p_cmb_guess = np.mean(gravity[i_cmb + 1:]) * rho_m_av * (
            Rp - Rc)  # not used but neat that it can be not far off
    p_mantle_bar = pressure[i_cmb + 1:][::-1] * 1e-5  # convert Pa to bar and invert
    T_mantle = temperature[i_cmb + 1:][::-1]
    print('initial guesses | p_cen =', pressure[0] * 1e-9, 'GPa | p_cmb =', p_cmb * 1e-9, 'GPa | Rp =', Rp / p.R_E,
          'R_E')

    # Iteration
    print('>>>>>>>>>\nIterating interior structure...')
    it = 1
    iter_param_old = 1e-5
    iter_param = p_cmb  # Rp
    while (abs((iter_param - iter_param_old) / iter_param_old) > tol) and (it < maxIter):
        # store old value to determine convergence
        iter_param_old = iter_param

        for i in range(n):  # M: 1:n, P: 0:n-1  index from centre to surface
            if i <= i_cmb:  # get local thermodynamic properties - core - layer by layer
                if pressure[i] > 10e3 * 1e9:
                    print('pressure error, i', i, pressure[i] * 1e-9, 'GPa', 'rho cen', density[0], 'rho cmb',
                          density[i_cmb])
                    # plt.plot(radius * 1e-3, pressure * 1e-9)
                    # plt.axvline(radius[i_cmb] * 1e-3)
                    # plt.show()
                _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 4)

                if cp[i] == 0:
                    print('i', i, 'cp[i]', cp[i], 'problem with core EoS')
                    raise ZeroDivisionError

            else:
                if pressure[i] > ppv_:
                    print('pressure error, i', i, pressure[i] * 1e-9, 'GPa', 'rho cen', density[0], 'rho cmb',
                          density[i_cmb])
                    # plt.plot(radius * 1e-3, pressure * 1e-9)
                    # plt.axvline(radius[i_cmb] * 1e-3)
                    # plt.show()
                _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 4)

                if cp[i] == 0:
                    print('i', i, 'cp[i]', cp[i], 'problem with core EoS')
                    raise ZeroDivisionError

        # update mass and radius - 1st mass entry never changes
        for i in range(2, n + 1):  # goes to i = n-1
            mass[n - i] = mass[n - i + 1] - M / n  # such that it never goes to 0 (would cause numerical errors)
        radius[0] = 0  # np.cbrt(mass[0] / density[0] / (4 * np.pi / 3))
        # print('radius[0]', radius[0])
        for i in range(1, n):
            dmass = mass[i] - mass[i - 1]
            radius[i] = np.cbrt((dmass / density[i] / (4 * np.pi / 3)) + radius[i - 1] ** 3)

        gravity = eos.g_profile(n, radius, density)

        # pressure and temperature are interpolated from surface downwards
        pressure, temperature = eos.pt_profile(n, radius, density, gravity, alpha, cp, Psurf, Tp, i_cmb,
                                               deltaT_cmb)

        # i_cmb = np.argmax(radius > Rc)   # update index of top of core in profiles
        i_cmb = np.argmax(mass > Mc)
        Rp = radius[-1]
        Rc = radius[i_cmb]

        p_mantle_bar = pressure[i_cmb + 1:][::-1] * 1e-5  # convert Pa to bar
        T_mantle = temperature[i_cmb + 1:][::-1]

        # update parameter that should be converging
        iter_param = pressure[i_cmb]  # Rp

        it = it + 1
        if it == maxIter:
            print('WARNING: reached maximum iterations for interior solver')
        # end while
