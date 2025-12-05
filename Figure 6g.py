import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define fonts to prevent box display and increase font size
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25  # Default font size
plt.rcParams['axes.labelsize'] = 25  # Axis label font size
plt.rcParams['xtick.labelsize'] = 25  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 25  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 25  # Legend font size

z_fixed = 1 


def fuction(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
    x, y, z, w = Fx
    dxdt = (-1 + x) * x * (h * (-1 + z) * Fp + h * (-1 + y) * Ft - h * Lg +
                           z * Rp1 + Rp2 - z * Rp2 +
                           y * Rt1 + Rt2 - y * Rt2 - Sg +
                           z * Tg + y * alpha * Tg - z * alpha * Tg)
    dydt = (-1 + y) * y * (Cq1 - Cq2 + Cq3 - h * Dt - h * x * Ft -
                           Rt1 + Rt2 - x * alpha * Tg - Υt)
    dzdt = (-1 + z) * z * (Cp1 - Cp2 - h * Dp - h * x * Fp -
                           Rp1 + Rp2 - x * Tg +
                           x * alpha * Tg - Υp)
    dwdt = (-1 + w) * w * (-L * y * z + h * L * (-1 + y * z) -
                           (1 + x + 2 * y + 4 * z) * Rv)
    return dxdt, dydt, dzdt, dwdt  

fig = plt.figure()

ax = fig.gca(projection='3d')

t = np.arange(0, 50, 0.005) 

args = (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)


for x in np.arange(0.1, 1, 0.2): 
    for y in np.arange(0.1, 1, 0.2):  
        for w in np.arange(0.1, 1, 0.2):  
           
            track11 = odeint(fuction, (x, y, z_fixed, w), t, args) 
            
            ax.plot(track11[:, 0], track11[:, 1], track11[:, 3])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$y$", labelpad=10)
ax.set_zlabel(r"$w$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)


plt.show()





