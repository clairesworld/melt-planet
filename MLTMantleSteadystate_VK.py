""" after Vilella & Kamata 2022
mixing length 1D temperatures in steady state, to calculate supersolidus temperature distribution
"""
import MLTMantle as mlt
import melting_functions as melt
import numpy as np
from math import sqrt, log, log10, exp


# # dimensional constants
# default_attr = {
# 'rho': 3400,
# 'g': 3,
# 'alpha': 5e-5,
# 'd': 1000e3,  # depth in m
# 'k': 3,
# 'kappa': 7e-7,
# 'eta': 1e20,
# 'L': 600e3,
# 'Tsurf': 250,  # upper boundary condition temperature
# }

# RaH = 1e6
# # H = 4.81e-8  # for RaH = 1e6 (fig 8) - min melting
# H = 4.87e-8  # for RaH = 1e6 - max melting}


class MLTMantleSteadystate(mlt.MLTMantle):
    """ I think H is in W m-3 """

    def temperature_scale(self, H):
        """" temperature scale for nondimensionalisation """

        # internal heating
        return H * self.d ** 2 / self.k_m[-1]  # k is an array but want this to be a scalar

    def RayleighRoberts(self, H, eta):
        """ Rayleigh-Roberts number for internal heating
        requires H and eta which may vary - other parameters are constant with time and possibly with depth """
        dTH = H * self.d ** 2 / self.k_m
        return self.rho_m * self.g_m * self.alpha_m * dTH * self.d ** 3 / (self.kappa_m * eta)

    def get_mixing_length(self, z, RaH=None, delta_tbl=None, c=None, d=None, Nu_max=1.0):
        """ function to calculate value of mixing length at z """

        b = 1 - delta_tbl  # z* at max value
        a = (18 * Nu_max * b / (RaH * self.get_dTdz_ambient(None, b, c, d) ** 2)) ** (1 / 4)  # max value of l*
        if z < b:
            return np.maximum(a / b * z, 1e-5)  # distance from upper boundary
        else:
            return np.maximum(-a / (1 - b) * z + a + (a * b) / (1 - b), 1e-5)  # distance from lower boundary

    def get_dTdz_ambient(self, T, z, c=None, d=None):
        # nondimensional "ambient" T profile
        return c / z ** d

    def get_delta_tbl_avg(self, RaH):
        # nondimensional bdy layer thickness
        if RaH < 1e5:
            return 0.8241 * RaH ** -0.06637
        elif RaH < 1e7:
            return 4.4412 * RaH ** -0.2203
        elif RaH < 1e9:
            return 5.6995 * RaH ** -0.2352
        else:
            raise Exception('RaH out of bounds')

    def get_delta_tbl_hot(self, RaH):
        # nondimensional bdy layer thickness
        if RaH < 1e5:
            return 1.7881 * RaH ** -0.1558
        elif RaH < 1e7:
            return 4.3793 * RaH ** -0.234
        elif RaH < 1e9:
            return 4.6238 * RaH ** -0.2361
        else:
            raise Exception('RaH out of bounds')

    def build_steadystate_Tz(self, RaH, lp=None, c=None, d=None, Nu_max=1.0, H=None, eta=None):
        """ nondimensional steady state solution for pure internal heating, Vilella & Kamata (A1) """

        T = [0] * self.Nm  # upper boundary condition T=0

        if RaH is None:
            assert H is not None
            assert eta is not None
            RaH = self.RayleighRoberts(H, eta)

        # build steady-state T profile downwards
        for ii in range(self.Nm - 1, 0, -1):
            T2 = T[ii]
            z2 = self.zp[ii]
            z1 = self.zp[ii - 1]
            l = lp[ii]

            # from sympy solution
            dTdz = [(z1 ** d * (RaH * c * l ** 4 + 9 * z1 ** d) - 3 * sqrt(
                z1 ** (3 * d) * (2 * RaH * c * l ** 4 + 2 * RaH * l ** 4 * Nu_max * z1 ** (d + 1) + 9 * z1 ** d))) / (
                                RaH * l ** 4 * z1 ** (2 * d)),
                    (z1 ** d * (RaH * c * l ** 4 + 9 * z1 ** d) + 3 * sqrt(z1 ** (3 * d) * (
                            2 * RaH * c * l ** 4 + 2 * RaH * l ** 4 * Nu_max * z1 ** (d + 1) + 9 * z1 ** d))) / (
                                RaH * l ** 4 * z1 ** (2 * d))]

            # for now assume 1st solution is the -ve one and the other is +ve
            T[ii - 1] = T2 - dTdz[0] * self.m
        return T

    def solve(self, RaH, H):
        zp = self.zp
        dTH = self.temperature_scale(H)

        # mixing lengths and boundary layers
        d_tbl_avg = self.get_delta_tbl_avg(RaH)
        lavg = [self.get_mixing_length(z, RaH, d_tbl_avg, c=0.05, d=1.55) for z in zp]

        d_tbl_hot = self.get_delta_tbl_hot(RaH)
        lhot = [self.get_mixing_length(z, RaH, d_tbl_hot, c=0.055, d=1.52, Nu_max=1.65) for z in zp]

        # get Tavg and Thot
        Tavg = self.build_steadystate_Tz(RaH, lavg, c=0.05, d=1.55)
        Thot = self.build_steadystate_Tz(RaH, lhot, c=0.055, d=1.52, Nu_max=1.65)  # see above

        # get 95% hottest temperatures
        T95 = [melt.T_at_95(Ta, Th) for Ta, Th in zip(Tavg, Thot)]

        # define solidus temperatures
        Tsol = melt.T_solidus_pyrolite_extrap(self.P)

        Xmelt = melt.supersolidus_fraction(Tsol, T95, Thot, Tscale=dTH)

        return_dict = {'Tavg': Tavg, 'T95': T95, 'Thot': Thot, 'Xmelt': Xmelt}
        return return_dict