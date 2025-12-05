import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define fonts to prevent box display
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25  # Default font size
plt.rcParams['axes.labelsize'] = 25  # Axis label font size
plt.rcParams['xtick.labelsize'] = 25  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 25  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 25  # Legend font size


k = 0.5

# Define the system of ODEs
def fuction(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
    x, y, z, k = Fx
    dxdt = (-1 + x) * x * (h * (-1 + z) * Fp + h * (-1 + y) * Ft - h * Lg +
                           z * Rp1 + Rp2 - z * Rp2 +
                           y * Rt1 + Rt2 - y * Rt2 - Sg +
                           z * Tg + y * alpha * Tg - z * alpha * Tg)
    dydt = (-1 + y) * y * (Cq1 - Cq2 + Cq3 - h * Dt - h * x * Ft -
                           Rt1 + Rt2 - x * alpha * Tg - Υt)
    dzdt = (-1 + z) * z * (Cp1 - Cp2 - h * Dp - h * x * Fp -
                           Rp1 + Rp2 - x * Tg +
                           x * alpha * Tg - Υp)
    dkdt = (-1 + k) * k * (-L * y * z + h * L * (-1 + y * z) -
                           (1 + x + 2 * y + 4 * z) * Rv)
    return dxdt, dydt, dzdt, dkdt

# Setup the main plot
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Time or evolution step
t = np.arange(0, 50, 0.005)

# Define parameters except t, x, y, z
settings = [
    (5, 4.5, 3, 8, 5.6, 0.5, 6, 1.3, 0.1, 2, 3, 2, 0.03, 0.01, 2.42, 8, 2, 1.1, 2, 2, 1, 4),
    (5, 4.5, 5, 8, 5.6, 0.5, 6, 1.3, 0.5, 2, 3, 2, 0.03, 0.01, 2.42, 8, 2, 1.1, 2, 2, 1, 4),
    (5, 4.5, 7, 8, 5.6, 0.5, 6, 1.3, 0.8, 2, 3, 2, 0.03, 0.01, 2.42, 8, 2, 1.1, 2, 2, 1, 4),
    (5, 4.5, 9, 8, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 0.03, 0.01, 2.42, 8, 2, 1.1, 2, 2, 1, 4)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2, 0.2), t, args)
    ax.plot(track5[:, 0], track5[:, 1], track5[:, 2], color)

# Setting up main and subplot properties
ax.view_init(elev=20, azim=-130)
ax.set_facecolor('white')  # Set background color to white
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$y$", labelpad=10)
ax.set_zlabel(r"$z$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)

# Set tick label size for z-axis directly
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontsize(20)

plt.legend(labels=('$α$=0.1', '$α$=0.5', '$α$=0.8', '$α$=1'), loc='best')

plt.xticks(np.arange(0, 1.01, step=0.2))
plt.yticks(np.arange(0, 1.01, step=0.2))
ax.set_zticks(np.arange(0, 1.01, step=0.2))

# Add subplot
left, bottom, width, height = 0.26, 0.35, 0.2, 0.2
ax1 = fig.add_axes([left, bottom, width, height])
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2, 0.2), t, args)
    ax1.plot(track5[:, 0], track5[:, 2], color)  # Plot x versus y instead of z

ax1.set_facecolor('whitesmoke')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1, step=0.2), [])
plt.yticks(np.arange(0, 1, step=0.2), [])

plt.text(0.8, 0.1, s="x", transform=ax1.transAxes, fontsize=25)
plt.text(0.1, 0.8, s="z", transform=ax1.transAxes, fontsize=25)  # Update label to y
plt.text(0.46, 0.02, s="0", transform=ax.transAxes, fontsize=25)

plt.show()