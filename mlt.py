"""
Module to calculate mantle thermal evolution via Mixing Length Theory
Based on Vilella & Kamata (2021) and  Tachinami, Senshu, and Ida (2011)
"""

import numpy as np
import interior_structure as intr
import pde


def mixing_length(r, a=0.4109, b=0.5537):
    D = np.max(r)
    r_cmb = np.min(r)
    h = r - r_cmb
    r_break = h >= (D / 2) * b

    l = np.zeros_like(r)
    l[:r_break + 1] = a * h / b
    l[r_break + 1:] = a * (D - h) / (2 - b)
    return l


# radial density distribution - use Vinet EOS from TSI11 for now but can use Perplex later - constant over time
pl = intr.make_test_earth()
T0 = pl.temperature[43:]
z = pl.radii[43:]
T_sfc = np.min(T0)

""" SOLVE 1D PDE 
https://py-pde.readthedocs.io/_/downloads/en/stable/pdf/
"""

params_default = {
    'k_cond': 3,
    'rho': 3500,
    'alpha': 5e-5,
    'g': 9.8,
    'kappa': 7e-7,
    'Cp': 1260,
    'L': 600e3,
    # 'eta_0': 10 ** 20,
    # 'Ea':
    'l': mixing_length(z),
    'grad_adiabat': 10,
}


def viscosity(field, p=params_default):
    return 10 ** 20




class HeatTransferPDE(pde.PDEBase, p=params_default):

    def evolution_rate(self, field, t=0):
        """implement the python version of the evolution equation"""
        assert field.grid.dim == 1  # ensure the state is one-dimensional

        # define boundary conditions
        bc_upper = {"value": T_sfc}
        bc_lower = {"derivative": 0}
        bc = [bc_lower, bc_upper]
        grad_x = field.gradient(bc)[0]

        eta = viscosity(field, p)

        F_conv = p['rho'] ** 2 * p['Cp'] * p['alpha'] * p['g'] * p['l'] ** 4 / (18 * eta) * (grad_x - p['grad_adiabat']) ** 2

        dF_cond = p['k_cond'] * field.laplace(bc)[0]
        dF_conv =




        rhs =

        return 6 * field * grad_x - grad_x.laplace("auto_periodic_neumann")



# initialize the equation and the spac
grid = pde.CartesianGrid([[np.min(z), np.max(z)]], len(z), periodic=False)
state_ini = pde.ScalarField(grid, T0)

# solve the equation and store the trajectory
storage = pde.MemoryStorage()
eq = HeatTransferPDE()
eq.solve(state_ini, t_range=3, method="scipy", tracker=storage.tracker(0.1))

# plot the trajectory as a space-time plot
# plot_kymograph(storage)