import matplotlib.pyplot as plt
import pandas as pd
from MLTMantle import years2sec, get_Mantle_struct

def plot_benchmark_T(csv_file, file_path='output/tests/benchmarks/', label='benchmark', fig=None, ax=None):
    df = pd.read_csv(file_path + csv_file, names=['radius', 'temperature'])
    ax.plot(df.radius, df.temperature, label=label)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Temperature (K)')
    return fig, ax

def plot_benchmark_q(csv_file, file_path='output/tests/benchmarks/', label='benchmark', fig=None, ax=None):
    df = pd.read_csv(file_path + csv_file, names=['time', 'q_surf'])
    ax.plot(df.time, df.q_surf, label=label)
    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel('Surface heat flux (W/m2)')
    ax.set_yscale('log')
    return fig, ax


def plot_model_T(name, output_path='output/tests/tmp/', label='model', radius=None, fig=None, ax=None):
    import glob
    import pickle as pkl

    f = glob.glob(output_path + name + '.pkl*')[-1]  # get last file

    with open(f, "rb") as pfile:
        d = pkl.load(pfile)

    ax.plot(radius * 1e-3, d['u'], label=label)
    ax.set_xlim(radius[0] * 1e-3, radius[-1] * 1e-3)
    return fig, ax


def plot_model_q(name, output_path='output/tests/tmp/', label='model', fig=None, ax=None):
    import glob
    import pickle as pkl

    plot_key = 'q'
    files = glob.glob(output_path + name + '.pkl*')

    q, t = [], []
    for f in files:  # all timesteps
        with open(f, "rb") as pfile:
            try:
                d = pkl.load(pfile)
                q.append(d[plot_key][-1])
                t.append(d['t'] / years2sec * 1e-9)
            except EOFError as e:
                print(e, f)

    t, q = zip(*sorted(zip(t, q)))
    ax.plot(t, q, label=label)
    return fig, ax


def plot_adiabat(man, Tp, psurf=1000, fig=None, ax=None):
    from PlanetInterior import pt_profile
    n = len(man.r)

    P, T_ad = pt_profile(n, radius=man.r, density=man.rho_m, gravity=man.g_m, alpha=man.alpha_m, cp=man.cp_m,
                         psurf=psurf, Tp=Tp)

    ax.plot(man.r * 1e-3, T_ad, label=str(Tp) + 'K adiabat')
    return fig, ax



name = 'Tachinami_buffered2'
fig, axes = plt.subplots(1, 2)
man = get_Mantle_struct()

fig, axes[0] = plot_model_T(name, radius=man.r, label='5 Gyr model', fig=fig, ax=axes[0])
fig, axes[0] = plot_benchmark_T('Tachninami_TR_1ME_1Gyr.csv', label='1 Gyr T11', fig=fig, ax=axes[0])
fig, axes[0] = plot_benchmark_T('Tachninami_TR_1ME_5Gyr.csv', label='5 Gyr T11', fig=fig, ax=axes[0])
fig, axes[0] = plot_adiabat(man=man, Tp=1900, fig=fig, ax=axes[0])
axes[0].legend()

fig, axes[1] = plot_model_q(name, fig=fig, ax=axes[1])
fig, axes[1] = plot_benchmark_q('Tachninami_qt_1ME.csv', fig=fig, ax=axes[1])
axes[1].legend()

plt.show()



