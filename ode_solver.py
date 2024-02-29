from numpy import linspace, zeros, asarray, diff, maximum
import matplotlib.pyplot as plt
import numpy as np

years2sec = 3.154e7

def ode_FE(f, U_0, dt, T):
    N_t = int(round(float(T) / dt))
    # Ensure that any list/tuple returned from f_ is wrapped as array
    f_ = lambda u, t: asarray(f(u, t))
    u = zeros((N_t + 1, len(U_0)))
    t = linspace(0, N_t * dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n + 1] = u[n] + dt * f_(u[n], t[n])
    return u, t


def mixing_length(z):
    """ function to calculate value of mixing length at z - nondimensional
     I think Ra_b is bottom thermal Ra? """
    alpha_mlt = 0.2895
    beta_mlt = 0.6794

    # Wagner eq (11)
    h = z  # dimensionless height above CMB
    D = 1  # dimensionless thickness of entire mantle is 1 by construction
    if h <= D / 2 * beta_mlt:
        return maximum(alpha_mlt * h / beta_mlt, 1e-5)
    else:
        return maximum(alpha_mlt * (D - h) / (2 - beta_mlt), 1e-5)


def chi(alpha, rho, cp, gs, l, eta):
    # time scale (from W/K2)
    X = alpha * rho ** 2 * cp * gs * l ** 4 / (18 * eta)
    return X


def adiabat(u, alpha, cp, gs):
    return -alpha / cp * gs * u  # K/m


def rhs(u, t):
    # build down du/dt from x=1 @ t
    N = len(u) - 1
    dudt = zeros(N + 1)
    dudt[-1] = dsdt(t)  # value of du/dt at x=L - i.e. constant temperature so no dT/dt
    A = chi(alpha, rho, cp, gs, l, eta) * Theta * Theta ** 2  # advection term prefactor
    dAdx = diff(A) / dx  # size N
    dTdz_ad = adiabat(u, alpha, cp, gs)
    dT2dz2_ad = diff(dTdz_ad) / dx
    kp = k * Theta * Theta ** 2

    for i in range(N-1, 0, -1):

        d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
        dudx = (u[i + 1] - u[i]) / dx
        print('dudx', dudx)

        if abs(dTdz_ad[i]) > abs(dudx):  # -ve kv
            dudt[i] = kp * d2udx2
            print('              kv=0 at', i, '/', N)

        else:
            dudt[i] = (kp * d2udx2 - dAdx[i] * (dudx ** 2 - 2 * dudx * dTdz_ad[i] + dTdz_ad[i] ** 2)
                       - A[i] * (2 * dudx * d2udx2 - 2 * d2udx2 * dTdz_ad[i] - 2 * dudx * dT2dz2_ad[i] + 2 * dTdz_ad[i] * dT2dz2_ad[i])
                       + g(x[i], t))
    # dudt[1:N - 1] = (beta / dx ** 2) * (u[2:N + 1] - 2 * u[1:N] + u[0:N - 1]) + g(x[1:N], t)  # faster
    # dudt[N] = (beta / dx ** 2) * (2 * u[N - 1] + 2 * dx * dudx(t) - 2 * u[N]) + g(x[N], t)
    dudt[0] = (kp / dx ** 2) * (2 * u[0] + 2 * dx * du0dx(t) - 2 * u[1]) + g(x[1], t)  # cheat to say only conduction at lower boundary
    return dudt * rho * cp


def rhs2(u, t):
    # build down du/dt from x=1 @ t
    N = len(u) - 1
    dudt = zeros(N + 1)
    dudt[-1] = dsdt(t)  # value of du/dt at x=L - i.e. constant temperature so no dT/dt
    # A = chi(alpha, rho, cp, gs, l, eta) * Theta * Theta ** 2  # advection term prefactor
    # dAdx = diff(A) / dx  # size N
    dTdz_ad = adiabat(u, alpha, cp, gs)
    # dT2dz2_ad = diff(dTdz_ad) / dx
    # kp = k * Theta * Theta ** 2
    kc = k

    q = np.zeros_like(dudt)
    for i in range(N - 1, 0, -1):
        d2udx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
        dudx = (u[i + 1] - u[i]) / dx
        # print('dudx', dudx)

        if abs(dTdz_ad[i]) > abs(dudx):  # -ve kv
            kv = 0
        else:
            kv = alpha * rho ** 2 * cp * gs * l[i] ** 4 / (18 * eta) * (dTdz_ad[i] - dudx)

        conv_part = -kc * dudx
        cond_part = -kv * (dudx - dTdz_ad[i])
        q[i] = cond_part + conv_part

    # print('q', np.shape(q), q)
    gradq = diff(q) / dx
    # print('grad q', np.shape(gradq), gradq)
    dudt[1:N] = -gradq[1:N] + g(x, t)

    # cheat to say only conduction at lower boundary
    dudt[0] = (kc / dx ** 2) * (2 * u[0] + 2 * dx * du0dx(t) - 2 * u[1]) + g(x[1], t)
    return dudt * rho * cp


def du0dx(t):
    # bottom boundary condition flux (originally x=L)
    return 0


def s(t):
    # upper boundary condition temperature (originally x=0)
    return Tsurf


def dsdt(t):
    return 0


def g(x, t):
    # source term
    return rho * H


def initial(z):
    return (Tsurf-Tcmb0) * z + Tcmb0


def test_diffusion_advection():
    # solve with dimensionless length and time
    global Tsurf, Tcmb0, dx, L, Theta, x, Nm, rho, H, k, gs, alpha, cp, eta, l  # needed in rhs
    L = 3000e3  # length scale in m
    Theta = np.pi ** 7  #4e9 * years2sec  # time scale in s
    tmax = 1 * Theta
    Tsurf = 300
    Tcmb0 = 2850

    eta = 1e20
    cp = 1190
    alpha = 3e-5
    gs = 10
    k = 5
    rho = 4500  #* L ** 3
    H = 1e-12  #/ L ** 2 * tmax ** 2  # kg·m²/s²/kg

    Nm = 5000
    Nt = 2000

    x = linspace(0, L, Nm + 1)
    dx = x[1] - x[0]

    U_0 = zeros(Nm + 1)
    U_0[-1] = s(0)  # upper fixed T
    U_0[:-1] = initial(x[:-1]/L)  # initial temperature

    plt.plot(x / L, U_0)
    plt.xlabel('x/L')
    plt.ylabel('u(x,0)')
    plt.legend(['t=0'])
    plt.show()

    dt = tmax / Nt
    print('dt', dt)

    l = np.array([mixing_length(zz) * L for zz in x])
    u, t = ode_FE(rhs2, U_0, dt, T=tmax)

    return u, t, x


u_solution, t_solution, x_solution = test_diffusion_advection()
print(np.shape(u_solution))

i = 1
y = u_solution[i,:]
lines = plt.plot(x_solution/L, y)
plt.xlabel('x/L')
plt.ylabel('u(x,t)')
plt.legend(['t=%.0f' % (t_solution[i] / years2sec)])
plt.show()
