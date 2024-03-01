from HeatTransferSolver import test_arrhenius_radheating

test_arrhenius_radheating(N=1000, Nt_min=1000, t_buffer_Myr=3000, age_Gyr=4.5,
                          writefile='output/tests/radheating_fixedflux.h5py', plot=True,
                          figpath='figs_scratch/radheating_fixedflux.pdf')