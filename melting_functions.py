import numpy as np
from math import exp

def T_solidus_pyrolite(P):
    # Simon & Glatzel equation (1929), fit by Pierru+ 2022 (Table 3)
    if (P >= 5) and (P <= 24):  # GPa
        a = 61.118
        c = 1.1288
        T0 = 1609
    elif (P > 24) and (P < 139):
        a = 0.0300
        c = 3.0067
        T0 = 235
    else:  # out of bounds
        T0 = a = c = np.nan
    return T0 * (P/a + 1) ** (1/c)


def T_solidus_pyrolite_extrap(P):
    # Simon & Glatzel equation (1929), fit by Pierru+ 2022 (Table 3)
    # use fit between 5 and 24 GPa, assume extrapolates to surface
    a = 61.118
    c = 1.1288
    T0 = 1609
    return T0 * (P/a + 1) ** (1/c)


def T_solidus_chondrite(P):
    # Andrault+ 2011
    a = 92
    c = 1.3
    T0 = 2045
    return T0 * (P/a + 1) ** (1/c)


def T_liquidus_chondrite(P):
    # same
    a = 29
    c = 1.9
    T0 = 1940
    return T0 * (P/a + 1) ** (1/c)


def T_solidus_H2000(P):
    # peridotite solidus from Hirschmann 2000 Table 2, P in GPa
    a = -5.1404654
    b = 132.899012
    c = 1120.66061
    return a * P ** 2 + b * P + c


def T_at_95(Tavg, Thot, n95=1 / 3):
    # T profile corresponding to cdf of 95%
    return n95*Tavg + (1 - n95)*Thot


def cdf_hot(T_of_interest, T95, Thot, p=5):
    # nondimensional cumulative distribution of 95-100% hottest temperatures
    Trsc = (T_of_interest - T95)/(Thot - T95)  # rescale to 0, 1
    return np.maximum(95 + 5 * (1 - exp(-p * Trsc)), 0)  # cannot be negative


def supersolidus_fraction(Tsol, T95, Thot, Tscale):
    # find proportion of temperatures larger than solidus profile

    Xmelt = [] # proportion of samples at given depth with T > Tsol (%)
    for n in range(0, len(Thot)):
        Tsol_nondim = (Tsol[n] - Thot[-1]) / Tscale  # nondimensionalised solidus

        # maximum Xmelt is 5% - reasonable if melting regulates temperature
        if T95[n] > Tsol_nondim:
            percent_hotter = 5  # max 5% of T samples are supersolidus
        else:
            try:
                percent_hotter = 100 - cdf_hot(Tsol_nondim, T95[n], Thot[n])

            except ZeroDivisionError as e:
                 # i.e. solidus not defined -- P too small
    #             print('zero division error at' , n)
    #             print(Tsol_nondim, T95[n], Thot[n])
                percent_hotter = 0

        Xmelt.append(percent_hotter/100)

    return Xmelt
