import numpy as np
import math
from parameters import M_E, R_E, G

""" Calcualte planet interior structure self-consistently 
these methods result in layers with constant delta_mass.... todo constant radius so don't need to interpolate?
"""

P_ppv_in = 1183750e5  # Pa
P_pv_in = 227556e5  # Pa


def radius_from_bulkdensity(CMF, Mp, rho_c, rho_m):  # from bulk i.e. average values - just for initial guess
    if CMF > 0:
        Rc = (3 * CMF * Mp / (4 * math.pi * rho_c)) ** (1 / 3)
    else:
        Rc = 0
    Rp = (Rc ** 3 + 3 * (1 - CMF) * Mp / (4 * math.pi * rho_m)) ** (1 / 3)
    return Rc, Rp


def g_profile(n, radius, density):
    # gravity has boundary condition 0 in center (and not at surface)
    # analogously, a surface gravity can be determind from M and
    # guessed R, and gravity can be interpolated from surface downwards
    # Problem with that approach: neg. gravity values in center possible
    gravity = np.zeros(n)
    for i in range(1, n):
        gravity[i] = (radius[i - 1] ** 2 * gravity[i - 1] + 4 * np.pi * G / 3 * density[i] * (
                radius[i] ** 3 - radius[i - 1] ** 3)) / radius[i] ** 2
    return gravity


def adiabatic_lapse_rate(T, x, alpha, cp, gravity):
    # todo: does this need to be multiplied by dP/dz?
    return -alpha / cp * gravity * T


def pt_profile(n, radius, density, gravity, alpha, cp, psurf, Tp, i_cmb=None, deltaT_cmb=0):
    """ input psurf in bar, pressure and (adiabatic) temperature are interpolated from surface downwards """
    pressure = np.zeros(n)
    temperature = np.zeros(n)
    pressure[-1] = psurf * 1e5  # surface pressure in Pa
    temperature[-1] = Tp  # potential surface temperature, not real surface temperature
    for i in range(2, n + 1):  # M: 1:n-1; n-i from n-1...1; P: from n-2...0 -> i from 2...n
        dr = radius[n - i + 1] - radius[n - i]
        pressure[n - i] = pressure[n - i + 1] + dr * gravity[n - i] * density[n - i]
        lapse_rate = adiabatic_lapse_rate(temperature[n - i + 1], None, alpha[n - i], cp[n - i], gravity[n - i])
        temperature[n - i] = temperature[n - i + 1] - dr * lapse_rate
        if n - i == i_cmb:
            # add temperature jump across cmb (discontinuity)
            temperature[n - i] = temperature[n - i] + deltaT_cmb
    return pressure, temperature  # Pa, K


def pressure_lookup(z_of_interest, pressure_list):
    """ get pressure from dimensionless depth """
    z_list = list(np.linspace(0, 1, len(pressure_list)))
    # find pressure in pressure_list closest to z target
    return pressure_list[z_list.index(z_of_interest)]


def loadinterior(filepath):
    import pickle as pkl
    with open(filepath, "rb") as pfile:
        return pkl.load(pfile)


class PlanetInterior:

    def __init__(self, name='default', M=M_E, CMF=0.325, Psurf=1000, Tp=1600, deltaT_tbl=1000, deltaT_cmb=0,
                 verbose=False):
        self.mass = None
        self.dTdz_adiabat = None
        self.i_cmb = None
        self.Rc = None
        self.R = None
        self.cp = None
        self.alpha = None
        self.density = None
        self.temperature = None
        self.pressure = None
        self.radius = None
        self.gravity = None
        self.name = name
        self.M = M
        self.CMF = CMF
        self.Psurf = Psurf
        self.Tp = Tp
        self.deltaT_tbl = deltaT_tbl
        self.deltaT_cmb = deltaT_cmb

    def initialise_constant(self, n=50000, rho=None, cp=1300, alpha=2.5e-5, cp_c=800, alpha_c=1e-5):

        M = self.M
        CMF = self.CMF
        Mc = self.M * self.CMF
        Psurf = self.Psurf
        Tp = self.Tp
        deltaT_cmb = self.deltaT_cmb
        cp_m = cp
        alpha_m = alpha

        # Initial guess for planet radius assuming constant mantle and core densities
        x_Fe = CMF  # this is just for starting guess on Rp from parameterisation
        Rp = 1e3 * (7030 - 1840 * x_Fe) * (
                M / M_E) ** 0.282  # initial guesss, Noack & Lasbleis 2020 (5) ignoring mantle Fe
        if CMF > 0:
            Rc = 1e3 * 4850 * x_Fe ** 0.328 * (M / M_E) ** 0.266  # initial guess, hot case, ibid. (9)
            rho_c_av = x_Fe * M / (4 / 3 * np.pi * Rc ** 3)
        else:
            Rc = 0
            rho_c_av = 0
        if rho is None:
            rho_m_av = (1 - x_Fe) * M / (4 / 3 * np.pi * (Rp ** 3 - Rc ** 3))  # Noack & Lasbleis parameterisation
        else:
            rho_m_av = rho  # use initial guess as given
        Rc, Rp = radius_from_bulkdensity(CMF, M, rho_c_av, rho_m_av)  # get consistent radius

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

        gravity = g_profile(n, radius, density)
        pressure, temperature = pt_profile(n, radius, density, gravity, alpha, cp, Psurf, Tp, i_cmb, deltaT_cmb)
        p_cmb = pressure[i_cmb]  # the pressure (at cell top edge) of the cell that Rc passes through

        # store full planet values
        self.radius = radius
        self.mass = mass
        self.gravity = gravity
        self.pressure = pressure
        self.temperature = temperature
        self.density = density
        self.alpha = alpha
        self.cp = cp
        self.dTdz_adiabat = [adiabatic_lapse_rate(temperature[i], None, alpha[i], cp[i], gravity[i]) for i in range(0, n)]
        self.R = Rp
        self.Rc = Rc
        self.i_cmb = i_cmb  # say this is the top layer of the core

        print('initial guesses | p_cen =', '{:.4f}'.format(pressure[0] * 1e-9), 'GPa | p_cmb =',
              '{:.4f}'.format(p_cmb * 1e-9), 'GPa | Rp =', '{:.4f}'.format(Rp / R_E),
              'R_E | ToM density =', '{:.4f}'.format(density[-1]), 'kg m-3')

    def solve(self, maxIter=100, tol=0.0001):
        """ tweaked from Noack+
            tol is fractional convergence criterion for iteration solver

            first need to run initialise_constant() to get initial profiles
            """

        import eos

        if self.density is None:
            raise AssertionError('Interior structure not initialised (run initialise_constant())')

        # pre-defined values
        Psurf = self.Psurf
        Tp = self.Tp
        deltaT_cmb = self.deltaT_cmb
        n = len(self.pressure)

        # from initialisation
        mass = self.mass
        gravity = self.gravity
        density = self.density
        temperature = self.temperature
        pressure = self.pressure
        radius = self.radius
        alpha = self.alpha
        cp = self.cp
        M = self.M
        Mc = self.M * self.CMF
        R = self.R
        Rc = self.Rc
        rho_m_av = self.density[-1]
        i_cmb = self.i_cmb
        p_cmb = self.pressure[i_cmb]

        p_cmb_guess = np.mean(gravity[i_cmb + 1:]) * rho_m_av * (R - Rc)  # not used but neat that it can be not far off

        # Iteration
        print('>>>>>>>>>\nIterating interior structure...')
        it = 1
        iter_param_old = 1e-5
        iter_param = p_cmb  # Rp
        while (abs((iter_param - iter_param_old) / iter_param_old) > tol) and (it < maxIter):
            # store old value to determine convergence
            iter_param_old = iter_param

            # get new thermodynamic parameters from eos
            for i in range(n):  # M: 1:n, P: 0:n-1  index from centre to surface
                # print('layer', i, '/',n)
                if i <= i_cmb:  # get local thermodynamic properties - core - layer by layer
                    _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 4)
                elif pressure[i] > P_ppv_in:  # Mg-postperovskite
                    _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 3)
                elif pressure[i] > P_pv_in:  # Mg-perovskite
                    _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 2)
                else:  # forsterite
                    _, density[i], alpha[i], cp[i] = eos.EOS_all(pressure[i] * 1e-9, temperature[i], 1)

                if cp[i] == 0:
                    print('i', i, 'cp[i]', cp[i], 'problem with silicate EoS')
                    raise ZeroDivisionError

            # update mass and radius - 1st mass entry never changes
            for i in range(2, n + 1):  # goes to i = n-1
                mass[n - i] = mass[n - i + 1] - M / n  # such that it never goes to 0 (would cause numerical errors)
            radius[0] = 0  # np.cbrt(mass[0] / density[0] / (4 * np.pi / 3))
            # print('radius[0]', radius[0])
            for i in range(1, n):
                dmass = mass[i] - mass[i - 1]
                radius[i] = np.cbrt((dmass / density[i] / (4 * np.pi / 3)) + radius[i - 1] ** 3)

            gravity = g_profile(n, radius, density)

            # pressure and temperature are interpolated from surface downwards
            pressure, temperature = pt_profile(n, radius, density, gravity, alpha, cp, Psurf, Tp, i_cmb,
                                               deltaT_cmb)

            i_cmb = np.argmax(mass > Mc)

            # update parameter that should be converging
            iter_param = pressure[i_cmb]  # Rp

            it = it + 1

            print(it, 'p_cmb', iter_param)

            if it == maxIter:
                print('WARNING: reached maximum iterations for interior solver')
            # end while

        # store full planet values
        self.mass = mass
        self.radius = radius  # m
        self.pressure = pressure  # Pa
        self.temperature = temperature  # K
        self.density = density
        self.gravity = gravity
        self.alpha = alpha
        self.cp = cp
        self.dTdz_adiabat = [adiabatic_lapse_rate(temperature[i], None, alpha[i], cp[i], gravity[i]) for i in range(0, n)]
        self.R = R
        self.Rc = Rc

        # mantle start index
        self.i_cmb = i_cmb

    def plot_structure_r(self, c='k', fig_path='figs_scratch/', fig=None, ax=None, save=True, label=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=[7, 7])

        r_km = self.radius / 1000

        ax[0, 0].plot(self.pressure / 10 ** 9, r_km, color=c, alpha=1, label=label)
        ax[0, 0].set_xlabel("pressure (GPa)")
        ax[0, 0].set_ylabel("radius (km)")

        ax[0, 1].plot(self.temperature, r_km, color=c, alpha=1)
        ax[0, 1].set_xlabel("adiabatic temperature (K)")
        ax[0, 1].set_ylabel("radius (km)")

        ax[1, 0].plot(self.mass * 1e-24, r_km, color=c, alpha=1)
        ax[1, 0].set_xlabel("mass ($10^{24}$ kg)")
        ax[1, 0].set_ylabel("radius (km)")

        ax[1, 1].plot(self.density, r_km, color=c, alpha=1)
        ax[1, 1].set_xlabel("density (kg m$^{-3}$)")
        ax[1, 1].set_ylabel("radius (km)")

        plt.suptitle(self.name)

        for axx in ax.flatten():
            axx.axhline(self.radius[self.i_cmb] / 1000, ls='--', lw=0.5, c='k')
            axx.set_ylim(np.min(r_km), np.max(r_km))

        # ax[0, 0].legend(frameon=False)

        plt.tight_layout()

        if save:
            fig.savefig(fig_path + self.name + '_structure_r.pdf', bbox_inches='tight')
        return fig, ax

    def plot_structure_p(self, c='k', fig_path='figs_scratch/', fig=None, ax=None, save=True, label=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(2, 2, figsize=[7, 7])

        r_km = self.radius / 1000
        p_gpa = self.pressure / 10 ** 9

        ax[0, 0].plot(r_km, p_gpa, color=c, alpha=1, label=label)
        ax[0, 0].set_ylabel("pressure (GPa)")
        ax[0, 0].set_xlabel("radius (km)")

        ax[0, 1].plot(self.temperature, p_gpa, color=c, alpha=1)
        ax[0, 1].set_xlabel("adiabatic temperature (K)")
        ax[0, 1].set_ylabel("$P$ (GPa)")

        ax[1, 0].plot(self.mass * 1e-24, p_gpa, color=c, alpha=1)
        ax[1, 0].set_xlabel("$m$ ($10^{24}$ kg)")
        ax[1, 0].set_ylabel("$P$ (GPa)")

        ax[1, 1].plot(self.density, p_gpa, color=c, alpha=1)
        ax[1, 1].set_xlabel("density (kg m$^{-3}$)")
        ax[1, 0].set_ylabel("$P$ (GPa)")

        plt.suptitle(self.name)

        for axx in ax.flatten():
            axx.axhline(p_gpa[self.i_cmb], ls='--', lw=0.5, c='k')
            axx.set_ylim(np.max(p_gpa), np.min(p_gpa))

        # ax[0, 0].legend(frameon=False)

        plt.tight_layout()

        if save:
            fig.savefig(fig_path + self.name + '_structure_p.pdf', bbox_inches='tight')
        return fig, ax

    def save(self, output_path):
        import pickle as pkl
        with open(output_path + self.name + '_struct.pkl', "wb") as pfile:
            pkl.dump(self, pfile)

