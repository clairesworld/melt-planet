# plot internal heating
from HeatTransferSolver import rad_heating_forward
import matplotlib.pyplot as plt
import numpy as np
from MLTMantle import years2sec, get_Mantle_struct

man = get_Mantle_struct(struct_file='Tachinami_struct.pkl', output_path='output/tests/')
# this one is 1/2 too low but should be easier for getting too hot

plt.figure()
t = np.linspace(0, 5e9 * years2sec)  # seconds
print('tf', t[-1], 'seconds')

V_man = [4/3 * np.pi * (t ** 3 - b ** 3) for b, t in zip(man.r, man.r[1:])]
# H = [rad_heating_forward(tt, x=None, rho=man.rho_m) for tt in t]
H_TW = [rad_heating_forward(tt, x=None, rho=man.rho_m) * v * 1e-12 for tt, v in zip(t, V_man)]
# print(np.shape(H), 'H')

# H_TW = [h * v * 1e-12 for h, v in zip(H, V_man)]
print(np.shape(H_TW), 'H_TW', np.shape(H_TW[0]), 'H_TW[0]')
print(np.shape(V_man), 'V_man')

H_TW_tot = np.sum(np.array(H_TW), axis=1)
print(np.shape(H_TW_tot), 'H_TW_tot')

plt.plot(t / (years2sec * 1e9), H_TW_tot, label='Radiogenic heating')
plt.xlabel('t (Gyr)')
plt.ylabel('H (TW)')
#plt.ylabel('H (W/m3)')
plt.legend()
plt.show()