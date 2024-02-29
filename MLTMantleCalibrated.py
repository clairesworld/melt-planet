""" after Wagner+ 2019
mixing length calibrated to 3D convection models
"""
import MLTMantle as mlt
import melting_functions as melt
import numpy as np
from math import sqrt, log, log10, exp


def get_mixing_length_calibration(RaH, dEta):
    """ stagnant lid mixed heated, Wagner+ Table 4"""
    a0, a1, a2, a3 = 0.794, 0.132, -0.0515, 1.897
    b0, b1, b2 = 1.167, -0.0686, 0.129
    Ra_crit = 10 ** (4.323 + 2.922 * np.tanh(0.243 * np.log10(dEta) - 0.544))
    alpha_mlt = (a0 - a1 * np.log10(dEta) - a2 * np.log10(RaH)) * np.tanh(a3 * np.log10(RaH/Ra_crit))
    beta_mlt = b0 - b1 * np.log10(dEta) - b2 * np.log10(RaH)
    return alpha_mlt, beta_mlt


class MLTMantleCalibrated(mlt.MLTMantle):
    """" H is in W kg-1 """

    def temperature_scale(self, H):
        """" temperature scale for nondimensionalisation """
        # does it make sense for this to have a different value at each z
        # internal heating
        return H * self.rho_m[-1] * self.d ** 2 / self.k_m[-1]  # k is an array but want this to be a scalar

    def get_mixing_length_and_gradient(self, z, alpha_mlt, beta_mlt, l_is_smooth=True, **kwargs):
        """ function to calculate value of mixing length and its 1st derivative at z - nondimensional """
        try:
            assert l_is_smooth
        except AssertionError:
            raise NotImplementedError("Use smoothed mixing length or risk being numerically unhinged")

        # values for Ra = 1e7 and dEta = 1e6
        # for now these parameters are as in Wagner but might not be generalisable?
        alpha_mlt = 0.2895
        beta_mlt = 0.6794

        return mlt.get_mixing_length_and_gradient_smooth(z, alpha_mlt, beta_mlt, l_smoothing_distance=0.05, **kwargs)

    def get_internal_heating_rate(self, t, H=1e-12):
        # internal heating in W/m3, temp fixed H in W/kg
        return self.rho_m * H

    def get_viscosity(self, T, P, Tcmb=None, RaH=None, dEta=None, H=None):
        """ find viscosity at (dimensional) temperature given viscosity contrast and Ra """
        return mlt.exponential_viscosity_law(T, None, self.alpha_m, self.rho_m, self.gravity, self.d, self.kappa_m, #
                                         self.k_m, self.Tsurf, Tcmb, RaH, dEta, H)

    def build_steadystate_Tz(self, RaH, dEta=None, H=None, **kwargs):
        # from scipy.integrate import odeint

        lp = [self.get_mixing_length(z, Ra_b=None, dEta=None) for z in self.zp]  # dimensionless mixing length
        C = 0  # boundary condition for dTdz(z=0) = 0
        # T = [0] * self.Nm  # upper boundary condition T=0
        T = [0] * (self.Nm - 1) + [self.Tsurf]  # list of placeholder 0s to surface
        delta_r = self.m * self.d

        # build steady-state T profile downwards
        for ii in range(self.Nm - 1, 0, -1):
            z1 = self.zp[ii] * self.d  # dimensional length
            rho = self.rho_m[ii]
            k = self.k_m[ii]

            l = lp[ii] * self.d  # dimensionalise mixing length
            eta = self.get_viscosity(T[ii], P=None, RaH=RaH, dEta=dEta, H=H)
            chi = self.alpha_m[ii] * self.rho_m[ii] ** 2 * self.cp_m[ii] * self.g_m[ii] * l ** 4 / (18 * eta)
            print('z = {:.2f}'.format(self.zp[ii]), '/1    T =', T[ii], 'K', 'eta =', eta, 'Pa s')
            print('              chi = ', chi)

            try:
                # from sympy solution
                dTdz_ad = self.get_dTdz_ambient(T, self.zp[ii])  # what adiabatic gradient would be at this T and z
                dTdz = [(2 * chi * dTdz_ad + k - sqrt(
                    4 * C * chi + 4 * H * chi * rho * z1 + 4 * chi * dTdz_ad * k + k ** 2)) / (
                                2 * chi),
                        (2 * chi * dTdz_ad + k + sqrt(
                            4 * C * chi + 4 * H * chi * rho * z1 + 4 * chi * dTdz_ad * k + k ** 2)) / (
                                2 * chi)]
                print('              dT/dz soln =', dTdz)

                # for now assume 1st solution is the -ve one and the other is +ve
                dTdz = dTdz[0]
            except ValueError as e:
                # math domain error? -ve root?
                print('              root', 4 * C * chi + 4 * H * chi * rho * z1 + 4 * chi * dTdz_ad * k + k ** 2)
                print('              4 * H * chi * rho * z1', 4 * H * chi * rho * z1)
                print('              4 * chi * dTdz_ad * k', 4 * chi * dTdz_ad * k)
                print('              dT/dz adiabat', dTdz_ad)
                if abs(dTdz_ad) > abs(dTdz):  # -ve kv
                    dTdz = (-C - rho * H * z1) / k
                    print('              kv=0 at', ii, '/', self.Nm)
                else:
                    raise e

            # condition for no convection:
            if abs(dTdz_ad) > abs(dTdz):  # previous element
                dTdz = (-C - rho * H * z1) / k
                print('              kv=0 at', ii, '/', self.Nm)

            # increment temperature
            T[ii - 1] = T[ii] - dTdz * delta_r

        # this doesn't reproduce exactly because need to implement constant-T boundary condition to get value for C != 0
        return T
    #
    # def build_steadystate_Tz_odeint(self, RaH, dEta=None, H=None, **kwargs):
    #     # https://cmps-people.ok.ubc.ca/jbobowsk/Python/html/Jupyter%20Second%20Order%20ODEs.html
    #     from scipy.integrate import odeint
    #
    #     def z_derivatives(x, t):
    #         """
    #         The strategy to solve a second-order differential equation using odeint() is to write the equation as a
    #         system of two first-order equations. This is achieved by first writing x[1]=˙z and x[0]=z.
    #         One of our first-order equations is the expression above and the other is simply ˙z=x[1].
    #         """
    #         # get value of profile at x
    #         xprime = x / self.d  # nondimensional
    #         n = int(xprime * (self.Nm - 1))  # get index -- z from 0 to 1
    #         rho = self.rho_m[n]
    #         k = self.k_m[n]
    #         dTdz_ad = self.get_dTdz_ambient(xprime)
    #
    #         return [x[1], ]
    #
    #     lp = [self.get_mixing_length(z, Ra_b=None, dEta=None) for z in self.zp]  # dimensionless mixing length
    #     raise NotImplementedError




