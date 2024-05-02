""" class to calculate mantle temperature profiles using mixing length theory """
import numpy as np
from math import sqrt, log10, log
import PlanetInterior

# /network/group/aopp/planetary/RTP037_GUIMOND_MANTMELT

years2sec = 3.154e7

# define default dimensional attributes and which are expected to be in 1D
# list_of_1D_attr = ['alpha_m', 'cp_m', 'rho_m', 'g_m', 'k_m', 'dTdz_adiabat_m', 'T_adiabat']
default_attr = {'Tsurf': 300,  # surface temperature (K)
                }


def smooth_piecewise(x, x_threshold, f_left, f_right, df_left, df_right, smoothing_distance=0.05):
    """
    make a linear piecewise function C1 smooth (continuous first derivatives) around threshold point
    credit to https://notebook.community/ESSS/notebooks/smooth_transition_between_analytic_functions
    """

    # function to find smoothed center
    # Simplifying f_center to be an equation like x * (a*x + b) + c:
    def get_f_center_coefficients(x0, x1, f_left, df_left, df_right):
        dx = x1 - x0
        df0 = df_left(x0)
        df1 = df_right(x1)
        a = 0.5 * (df1 - df0) / dx
        b = (df0 * x1 - df1 * x0) / dx
        c = f_left(x0) - (x0 * (a * x0 + b))
        return a, b, c

    # split up piecewise
    x_0 = x_threshold * (1 - smoothing_distance)  # left bound
    x_1 = x_threshold * (1 + smoothing_distance)  # right bound

    a, b, c = get_f_center_coefficients(x_0, x_1, f_left, df_left, df_right)

    def f_smooth(x):
        return np.piecewise(x, [x < x_0, (x_0 <= x) & (x <= x_1), x > x_1], [
            f_left,
            lambda m: m * (a * m + b) + c,
            f_right,
        ])

    def df_smooth(x):
        return np.piecewise(x, [x < x_0, (x_0 <= x) & (x <= x_1), x > x_1], [
            df_left,
            lambda m: 2 * a * m + b,
            df_right,
        ])

    return f_smooth(x), df_smooth(x)  # ensure this stays above 0


def get_mixing_length_and_gradient_smooth(z, alpha_mlt=0.82, beta_mlt=1, l_smoothing_distance=0.05, **kwargs):
    """ function to calculate value of mixing length at z - nondimensional
    for Tachninami 2011, beta=1 and alpha=0.82"""
    D = 1  # dimensionless mantle height

    # values for Ra = 1e7 and dEta = 1e6
    # for now these parameters are as in Wagner but might not be generalisable?
    # alpha_mlt = 0.2895
    # beta_mlt = 0.6794

    # singularity point where mixing length is maximum (effective distance to nearest boundary)
    z_threshold = D / 2 * beta_mlt

    # define original piecewise function for mixing length
    f_left = lambda x: alpha_mlt * x / beta_mlt
    f_right = lambda x: alpha_mlt * (D - x) / (2 - beta_mlt)
    df_left = lambda x: alpha_mlt / beta_mlt
    df_right = lambda x: -alpha_mlt / (2 - beta_mlt)

    l_smooth, dl_smooth = smooth_piecewise(z, z_threshold, f_left, f_right, df_left, df_right,
                                           smoothing_distance=l_smoothing_distance)
    # print('l smooth', type(l_smooth), 'dl smooth', type(dl_smooth))
    return np.maximum(l_smooth, 1e-5), dl_smooth  # ensure l > 0


def exponential_viscosity_law(T, P, dEta, Tsurf, Tcmb, alpha=None, rho=None, gravity=None, L=None, kappa=None, kc=None,
                              RaH=None, H=None, eta_b=None, **kwargs):
    """ find viscosity at (dimensional) temperature given viscosity contrast and Ra """
    # scale to T' given d' and deta
    # for fixed temperature contrast
    Tprime = (T - Tsurf) / abs((Tsurf - Tcmb))
    return eta_b * np.exp(np.log(dEta) * (1 - Tprime))


def Arrhenius_viscosity_law(T, P, eta_ref, T_ref, Ea, **kwargs):
    Rb = 8.314
    return eta_ref * np.exp(Ea / Rb * (T ** -1 - T_ref ** -1))


def Arrhenius_viscosity_law_pressure(T, P, **kwargs):
    # must be able to return iterable if T/P iterable
    # P in Pa
    # Tackley 2013 eq 2-4, Table 2
    Rb = 8.314

    def f(T, P, E0, V0, P_decay, eta0, T0=1600):
        V = V0 * np.exp(-P / P_decay)
        H = E0 + P * V
        H0 = E0
        return eta0 * np.exp((H / (Rb * T)) - (H0 / (Rb * T0)))

    eta = np.where(P < PlanetInterior.P_pv_in,  # condition
                   f(T, P, 300e3, 5e-6, np.inf, 1e21),  # upper mantle
                   f(T, P, 370e3, 3.65e-6, 200e9, 3e23)  # lower mantle
                   )

    # truncate
    eta = np.maximum(eta, 1e19)
    eta = np.minimum(eta, 1e40)
    return eta


# def Arrhenius_viscosity_law_pressure(T, P, **kwargs):
#     # must be able to return iterable if T/P iterable
#     # P in Pa
#     # Tachinami eq 15
#     Rb = 8.314
#
#     def f(T, P, B, n, E, V, ep):
#         return 1/2 * ( 1/(B ** (1/n)) * np.exp((E + P * V)/(n * Rb * T)) ) * ep ** ((1 - n)/n)
#
#     return np.where(P < PlanetInterior.P_pv_in,  # condition
#              f(T, P, 3.5e-15, 3.0, 430e3, 10e-6, 1e-15),  # upper mantle
#              f(T, P, 7.4e-17, 3.5, 500e3, 10e-6, 1e-15)  # lower mantle
#                     )


def constant_viscosity_law(T, P, eta0=1e20, **kwargs):
    return eta0


class MLTMantle:
    """ base class for mantle, methods can be overwritten with other methods
    before calculating anything need to call set_dimensional_attr() and make_grid()
    inherits interior structure w/ defined rho, g, alpha, cp, P, r - this will need to be nondimensionalised somehow
    defaults for isoviscous case and constant internal heating rate """

    def __init__(self, planet, Nm=10000, Nt=10000, dimensional_attr=default_attr, verbose=False, **kwargs):
        self.kappa_m = None
        assert isinstance(planet, PlanetInterior.PlanetInterior)

        self.verbose = verbose
        self.planet = planet
        self.g_m = None
        self.tp = None
        self.zp = None
        self.u = None
        self.m = None
        self.alpha_m = None
        self.cp_m = None
        self.rho_m = None
        self.k_m = None
        self.dTdz_adiabat_m = None  # this is the (static) adiabatic profile used to solve for internal structure
        self.T_adiabat = None
        self.P = None
        self.d = None
        self.r = None
        self.Nt = None
        self.Nm = None
        self.Tsurf = None

        # initialise grid and assign time-independent parameter values
        self.make_grid(Nm, Nt)
        self.set_dimensional_attr(dimensional_attr)
        self.interpolate_mantle_structure()
        self.set_thermal_conductivity_and_diffusivity(**kwargs)

    def set_dimensional_attr(self, new_attr):
        """ update default parameters """
        # default_attr.update((key, val) for key, val in new_attr.items())
        self.__dict__.update((key, val) for key, val in new_attr.items())
        if self.verbose:
            print('updated attributes:', new_attr)

    def set_thermal_conductivity_and_diffusivity(self, k_um=5, k_lm=5):
        try:
            assert self.zp is not None
        except AssertionError:
            raise AssertionError('Mantle grid not initialised. Try make_grid() to set grid and '
                                 'interpolate_mantle_structure() to set parameter values')
        if k_um == k_lm:
            self.k_m = np.ones_like(self.zp) * k_um
            if self.verbose:
                print('set constant mantle thermal conductivity:', k_um, 'W/m/K')
        else:
            # find lower mantle assuming constant pv_in pressure
            i_lm = (np.abs(self.P - PlanetInterior.P_pv_in)).argmin()

            # single value for upper mantle and single value for lower mantle
            self.k_m = np.zeros_like(self.zp)  # initilaise array
            self.k_m[:i_lm + 1] = k_lm  # lower mantle
            self.k_m[i_lm + 1:] = k_um  # upper mantle
            if self.verbose:
                print('set thermal conductivitities:', k_um, 'W/m/K in UM,', k_lm, 'W/m/K in LM')
        self.kappa_m = self.k_m / (np.asarray(self.rho_m) * np.asarray(self.cp_m))  # thermal diffusivity in K-1

    def make_grid(self, Nm, Nt):
        """ function to make nondimensional grid """
        # number of grid points
        # defines the fidelity of the sim
        # memory usage is O(Nm * Nt) < 1e9!
        self.Nm = Nm
        # self.Nt = Nt

        # spatial bounds and derived space step.
        zmin, zmax = 0, 1
        self.m = (zmax - zmin) / Nm

        # # time bounds and derived timestep
        # tmin, tmax = 0, 1
        # self.u = (tmax - tmin) / Nt

        # initialise arrays
        self.zp = np.linspace(zmin, zmax, num=Nm, endpoint=True)  # z prime from 0->1
        # self.tp = np.linspace(zmin, zmax, num=Nm, endpoint=True)  # t prime from 0->1

        if self.verbose:
            print('initialised dimensionless grid array with Nm =', Nm, 'elements in z and Nt =', Nt, 'elements in t')

    def interpolate_mantle_structure(self, interpolation_method='linear'):
        """ get dimensional mantle values, interpolate pressure-density profiles if smaller grid"""
        try:
            assert self.Nm is not None
        except AssertionError:
            raise AssertionError('Missing grid (call make_grid())')

        i_base = self.planet.i_cmb + 1

        # dimensional radius
        self.r = np.linspace(self.planet.radius[i_base], self.planet.radius[-1], num=self.Nm,
                             endpoint=True)  # z prime from 0->1
        self.d = self.r[-1] - self.r[0]

        # get dimensional mantle values
        self.P = self.planet.pressure[i_base:]  # Pa
        self.T_adiabat = self.planet.temperature[i_base:]  # K
        self.dTdz_adiabat_m = self.planet.dTdz_adiabat[i_base:]  # K/m
        self.rho_m = self.planet.density[i_base:]  # kg/m3
        self.g_m = self.planet.gravity[i_base:]  # m/s2
        self.cp_m = self.planet.cp[i_base:]
        self.alpha_m = self.planet.alpha[i_base:]  # /K
        if self.verbose:
            print('sliced dimensional mantle profiles from interior structure')

        if self.Nm != len(self.planet.radius[i_base:]):
            # need to interpolate parameters to new mantle grid
            to_interp = [self.P, self.T_adiabat, self.dTdz_adiabat_m, self.rho_m, self.g_m, self.cp_m, self.alpha_m]
            for ii in range(len(to_interp)):
                if interpolation_method == 'linear':
                    to_interp[ii] = np.interp(self.r, self.planet.radius[i_base:], to_interp[ii])  # xnew, x, y
                else:
                    raise NotImplementedError('Interpolation method ' + interpolation_method + ' not implemented')
            self.P, self.T_adiabat, self.dTdz_adiabat_m, self.rho_m, self.g_m, self.cp_m, self.alpha_m = to_interp
            if self.verbose:
                print('interpolated mantle profiles from', len(self.planet.radius[i_base:]), 'to', self.Nm, 'elements')

    def temperature_scale(self, **kwargs):
        """" temperature scale for nondimensionalisation """
        # internal heating
        # return self.H * self.d ** 2 / self.k
        # bottom heating
        # todo
        raise NotImplementedError("")

    def RayleighRoberts(self, H, eta):
        """ Rayleigh-Roberts number for mixed-mode heating
        requires H and eta which may vary - other parameters are constant with time and possibly with depth """
        dTH = H * self.d ** 2 / self.k_m
        return self.rho_m ** 2 * self.g_m * self.alpha_m * dTH * self.d ** 3 / (self.kappa_m * eta)

    def get_mixing_length_and_gradient(self, z, alpha_mlt, beta_mlt, l_is_smooth=True, **kwargs):
        """ function to calculate value of mixing length at z - nondimensional """
        # if self.mixing_length_mode == 'best':
        #     # single value, from Tachinami: l = 0.82D
        #     # need to find D
        try:
            assert l_is_smooth
        except AssertionError:
            raise NotImplementedError("Use smoothed mixing length or risk being numerically unhinged")
        return get_mixing_length_and_gradient_smooth(z, alpha_mlt, beta_mlt, l_smoothing_distance=0.05, **kwargs)

    def get_dTdz_ambient(self, T, z, **kwargs):
        """ "ambient" T profile at dimensionless z"""
        try:
            # get entire profile+
            assert np.size(self.alpha_m) == np.size(T)
            return -self.alpha_m / self.cp_m * self.g_m * T
        except AssertionError:
            # return value at single depth
            # this is perhaps wrong because it would need to squish to Nm-1?
            n = int(z * (self.Nm - 1))  # get index -- z from 0 to 1
            return -self.alpha_m[n] / self.cp_m[n] * self.g_m[n] * T

    def get_internal_heating_rate(self, t, H=1e-12):
        # internal heating in W/m3
        # temp fixed H in W/kg
        return self.rho_m * H

    def get_viscosity(self, T, P):
        return 1e20  # temp

    def build_steadystate_Tz(self, RaH, **kwargs):
        """" steady state solution """
        raise NotImplementedError

    def solve(self, t0_Gyr, tf_Gyr, t0_buffer_Gyr=0, Nt_min=1000,
              viscosity_law=None,
              internal_heating_function=None,
              # radiogenic_conentration_factor=1,
              mixing_length_kwargs=None,
              viscosity_kwargs=None,
              internal_heating_kwargs=None,
              max_step=1e6 * years2sec,
              show_progress=True, verbose=False, writefile=None):
        import HeatTransferSolver as hts

        # get necessary thermal evolution kwargs
        if viscosity_kwargs is None:
            viscosity_kwargs = {}
        if mixing_length_kwargs is None:
            mixing_length_kwargs = {}
        if internal_heating_kwargs is None:
            internal_heating_kwargs = {}

        # set up grid/domain
        N = self.Nm
        Rp, Rc = self.r[-1], self.r[0]
        zp = self.zp
        dx = self.m
        L = Rp - Rc  # length scale
        D = 1  # dimensionless length scale

        # thermodynamic paramters
        cp = self.cp_m
        alpha = self.alpha_m
        gravity = self.g_m
        kc = self.k_m
        rho = self.rho_m
        kappa = self.kappa_m

        # boundary conditions
        Tsurf = self.Tsurf
        Tcmb0 = self.Tcmb0  # only used for initial condition, bc is constant flux

        # pressure profile
        pressures = self.P
        Tp = self.T_adiabat

        # static mixing length
        lp, dldx = get_mixing_length_and_gradient_smooth(zp, **mixing_length_kwargs)
        l = lp * self.d  # dimensionalise

        # todo neater way for this:
        internal_heating_kwargs.update({'rho': rho})  # ensure density is in internal heating to get W/m3

        # initial temperature profile
        # U_0 = hts.initial_linear(zp, Tsurf, Tcmb0)  # initial temperature
        # U_0 = hts.initial_steadystate(z, dz, Tsurf, l, rho, alpha, cp, kc, gravity, eta, H, dTdz_ad)
        U_0 = hts.initial_file("Tachinami.h5py", outputpath="output/tests/")

        if t0_buffer_Gyr > 0:
            # run for long enough to reach a steady state at constant H
            print('running to', t0_buffer_Gyr,' Gyr to approximate steady state initial condition')
            # args for heating_rate_function in HeatTransferSolver
            # needs to match signature to heating_rate_function
            # in this case assume constant heating

            H0 = internal_heating_function(t=0, x=zp, **internal_heating_kwargs)  # get initial heating rate
            print('H0 =', H0, 'W/kg')
            internal_heating_kwargs_initial = {'H0': H0}
            ivp_args = (
                dx, zp, get_mixing_length_and_gradient_smooth, hts.dudx_ambient, viscosity_law,
                hts.internal_heating_constant, kc, alpha, rho, cp, gravity, L,
                mixing_length_kwargs, viscosity_kwargs, internal_heating_kwargs_initial, l,
                pressures)

            t0, tf = 0, t0_buffer_Gyr * 1e9 * years2sec  # seconds

            max_step = hts.get_max_step(t0, tf, max_step, Nt_min, verbose)
            soln = hts.solve_pde(t0, tf, U_0, hts.calc_total_heating_rate_numeric, ivp_args,
                                 verbose=verbose, show_progress=True, max_step=max_step, writefile=False,
                                 first_step=100,)
            U_0 = soln.y[:, -1]

        else:
            # solve from initial U_0
            U_0 = U_0

        # now the real run
        # args for heating_rate_function in HeatTransferSolver
        # needs to match signature to heating_rate_function
        ivp_args = (
            dx, zp, get_mixing_length_and_gradient_smooth, hts.dudx_ambient, viscosity_law,
            internal_heating_function, kc, alpha, rho, cp, gravity, L,
            mixing_length_kwargs, viscosity_kwargs, internal_heating_kwargs, l,
            pressures)

        t0, tf = t0_Gyr * 1e9 * years2sec, tf_Gyr * 1e9 * years2sec  # seconds

        soln = hts.solve_pde(t0, tf, U_0, hts.calc_total_heating_rate_numeric, ivp_args,
                             verbose=verbose, show_progress=True, max_step=max_step, writefile=writefile)

        return soln


    def plot_temperature_evol(self, soln, timestep='all', cmap='magma', figpath=None):
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
        if timestep == 'all':
            for n in np.arange(len(soln.t))[::100]:
                plt.plot(self.zp, soln.y[:, int(n)], c=c[n])
        elif isinstance(timestep, int):
            plt.plot(self.zp, soln.y[:, timestep], c=c[timestep])
        else:
            raise NotImplementedError
        plt.xlabel('z/L')
        plt.ylabel('T (K)')
        plt.colorbar(
            plt.gca().scatter(soln.t / years2sec * 1e-6, soln.t / years2sec * 1e-6, c=soln.t / years2sec * 1e-6,
                              cmap='magma', s=0), label='time (Myr)')
        # plt.gca().set_xlim(0, 1)
        # plt.gca().set_ylim(250, 4500)

        if figpath is not None:
            fig.savefig(figpath, bbox_inches='tight')
        # plt.show()


def read_h5py(fin, outputpath, verbose=True):
    # read and plot remote runs

    import h5py
    d = {}
    with h5py.File(outputpath + fin, "r") as f:
        #         # Print all root level object names (aka keys)
        #         # these can be group or dataset names
        #         print("Keys: %s" % f.keys())
        #         # get first object name/key; may or may NOT be a group
        #         a_group_key = list(f.keys())[0]

        #         # get the object type for a_group_key: usually group or dataset
        #         print(type(f[a_group_key]))

        #         # If a_group_key is a group name,
        #         # this gets the object names in the group and returns as a list
        #         data = list(f[a_group_key])

        #         # If a_group_key is a dataset name,
        #         # this gets the dataset values and returns as a list
        #         data = list(f[a_group_key])
        #         # preferred methods to get dataset values:
        #         ds_obj = f[a_group_key]      # returns as a h5py dataset object
        #         ds_arr = f[a_group_key][()]  # returns as a numpy array

        if verbose:
            s = 'loaded ' + fin + ' with datasets'
            for i, key in list(enumerate(f.keys())):
                s = s + ' ' + key + ' ' + str(np.shape(f[key])) + ','
            print(s)

        # get numpy arrays
        for i, key in list(enumerate(f.keys())):
            d[key] = f[key][()]

    # return dictionary
    return d


def save_h5py_solution(fout, soln, ivp_kwargs={}, meta_dict=None):
    import h5py
    from HeatTransferSolver import calc_thermal_state

    # full thermal state of solution
    q, g, eta = calc_thermal_state(soln.t, soln.y, **ivp_kwargs)

    with h5py.File(fout, "w") as hf:
        hf.create_dataset('time', data=soln.t, dtype=soln.t.dtype)
        hf.create_dataset('temperature', data=soln.y, dtype=soln.y.dtype)
        hf.create_dataset('heat_flux', data=q, dtype=q.dtype)
        hf.create_dataset('internal_heating_rate', data=g, dtype=g.dtype)
        hf.create_dataset('viscosity', data=eta, dtype=eta.dtype)

        # add all other ivp args
        for k, v in ivp_kwargs.items():
            if callable(k):
                hf.create_dataset(k, data=eval(v + '.__name__'))  # turn functions into strings
            else:
                hf.create_dataset(k, data=v, dtype=v.dtype)

        if meta_dict:
            # store planet dictionary attrs as hdf5 metadata
            hf.attrs.update(meta_dict)
            # to print metadata: print(hf1.attrs.keys())
