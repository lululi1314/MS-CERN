Figure 3a
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

h = 1
k = 0.1

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
    (8, 4.5, 1, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 2, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 4, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4)
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

plt.legend(labels=('$T_g$=1', '$T_g$=2', '$T_g$=3', '$T_g$=4'), loc='best')
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
Figure 3b
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
    (80, 0.02, 20.38, 60, 0.06, 0.1, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 30.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 40.38, 60, 0.06, 0.8, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 1, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
Figure 3c
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
    (80, 0.02, 20.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.001, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 30.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.005, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 40.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.008, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$F_t$=1', '$F_t$=3', '$F_t$=5', '$F_t$=7'), loc='best')
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
Figure 3d
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
    (80, 0.02, 20.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.001, 1),
    (80, 0.02, 30.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.005, 1),
    (80, 0.02, 40.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.008, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$F_p$=1', '$F_p$=3', '$F_p$=5', '$F_p$=7'), loc='best')
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
Figure 3e
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

h = 1
k = 0.1

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
    (80, 0.02, 20.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 30.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 40.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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

plt.legend(labels=('$T_g$=20.38', '$T_g$=30.38', '$T_g$=40.38', '$T_g$=50.38'), loc='best',fontsize=25)
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
Figure 3f
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
    (80, 0.02, 50.38, 60, 0.06, 0.1, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.8, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 1, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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

plt.legend(labels=('$α$=0.1', '$α$=0.5', '$α$=0.8', '$α$=1'), loc='best', fontsize=25)
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
Figure 3g
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.001, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.1, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 1, 0.02, 0.02, 0.01, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$F_t$=0.001', '$F_t$=0.01', '$F_t$=0.1', '$F_t$=1'), loc='best')
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
Figure 3h
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.1, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 1, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 10, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$F_p$=0.01', '$F_p$=0.1', '$F_p$=1', '$F_p$=10'), loc='best')
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
Figure 4a
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
    (80, 0.02, 20.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.001, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 30.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.005, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 40.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.008, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$Cq3$=1', '$Cq3$=3', '$Cq3$=5', '$Cq3$=7'), loc='upper right')
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
Figure 4b
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
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 6.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 8.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 10.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4)
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
# Set legend location to the upper right corner with normal font weight
plt.legend(labels=('$Cq1$=4.9', '$Cq1$=6.9', '$Cq1$=8.9', '$Cq1$=10.9'),
           loc='upper right')

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
Figure 4c
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
    (8, 4.5, 3, 5, 5.6, 0.5, 4, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 8, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 10, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4)
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
plt.legend(labels=('$Rt1$=4', '$Rt1$=6', '$Rt1$=8', '$Rt1$=10'), loc='upper right')
# Set tick label size for z-axis directly
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontsize(20)

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
Figure 4d
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 0.1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 0.5, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 0.8, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Cq2$=4.5', '$Cq2$=6.5', '$Cq2$=8.5', '$Cq2$=10.5'), loc='upper right')

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
Figure 4e
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.1, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 1, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 10, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
]

colors = ['r+', 'b-', 'g--', 'y--']
for args, color in zip(settings, colors):
    track5 = odeint(fuction, (0.2, 0.2, 0.2,0.2), t, args)
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
plt.legend(labels=('$Cq3$=0.01', '$Cq3$=0.1', '$Cq3$=1', '$Cq3$=10'), loc='upper right')

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
Figure 4f
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.3, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 3, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 10, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
# Set legend location to the upper right corner with normal font weight
plt.legend(labels=('$Cq1$=0.03', '$Cq1$=0.3', '$Cq1$=3', '$Cq1$=10'),
           loc='upper right')

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
Figure 4g
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.8, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 8, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 20, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Rt1$=0.08', '$Rt1$=0.8', '$Rt1$=8', '$Rt1$=20'), loc='upper right')
# Set tick label size for z-axis directly
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontsize(20)

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
Figure 4h
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 5, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 10, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 15, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Υt$=1', '$Υt$=5', '$Υt$=10', '$Υt$=15'), loc='upper right')

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
Figure 5a
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


k = 1

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
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 2.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 4.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 6.42, 8, 2, 1, 2, 2, 1, 4),
    (8, 4.5, 3, 5, 5.6, 0.5, 6, 1.3, 1, 2, 3, 2, 4.9, 1, 8.42, 8, 2, 1, 2, 2, 1, 4)
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
plt.legend(labels=('$Cp1$=2.42', '$Cp1$=20.42', '$Cp1$=40.42', '$Cp1$=60.42'), loc='upper left')

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
Figure 5b
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 5),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 10),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 50)
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
plt.legend(labels=('$Rp1$=4', '$Rp1$=40', '$Rp1$=60', '$Rp1$=80'), loc='best')

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
Figure 5c
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 1.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 10.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 50.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Υp$=2', '$Υp$=20', '$Υp$=40', '$Υp$=60'), loc='best')

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
Figure 5d
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


k = 1

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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 1.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 5.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 10.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Cp1$=0.61', '$Cp1$=1.61', '$Cp1$=5.61', '$Cp1$=10.61'), loc='upper left')

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
Figure 5e
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 5),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 10),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 50)
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
plt.legend(labels=('$Rp1$=1', '$Rp1$=5', '$Rp1$=10', '$Rp1$=50'), loc='upper left')

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
Figure 5f
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
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 1.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 10.02, 0.02, 0.01, 1),
    (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 50.02, 0.02, 0.01, 1)
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
plt.legend(labels=('$Υp$=0.02', '$Υp$=1.02', '$Υp$=10.02', '$Υp$=50.02'), loc='upper left')

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
Figure 6a
import numpy as np
from scipy.integrate import odeint # type: ignore
import matplotlib.pyplot as plt

# Define fonts to prevent box display and increase font size
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25  # Default font size
plt.rcParams['axes.labelsize'] = 25  # Axis label font size
plt.rcParams['xtick.labelsize'] = 25  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 25  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 25  # Legend font size

x = 1  


def function(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
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
    return [dxdt, dydt, dzdt, dwdt]  


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

t = np.arange(0, 50, 0.005) 

args = (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)


for j in np.arange(0.1, 1, 0.2):  
    for k in np.arange(0.1, 1, 0.2):  
        for w in np.arange(0.1, 1, 0.2):  
            
            track11 = odeint(function, (x, j, k, w), t, args)
         
            ax.plot(track11[:, 1], track11[:, 2], track11[:, 3])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$y$", labelpad=10)
ax.set_ylabel(r"$z$", labelpad=10)
ax.set_zlabel(r"$w$", labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


plt.show()
Figure 6b
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

y_fixed = 1 


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
    for z in np.arange(0.1, 1, 0.2): 
        for w in np.arange(0.1, 1, 0.2):  
            
            track11 = odeint(fuction, (x, y_fixed, z, w), t, args) 
          
            ax.plot(track11[:, 0], track11[:, 2], track11[:, 3])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$z$", labelpad=10)
ax.set_zlabel(r"$w$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)


plt.show()
Figure 6c
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
Figure 6d
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

w = 1 



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
        for z in np.arange(0.1, 1, 0.2): 
            
            track11 = odeint(fuction, (x, y, z, w), t, args) 
           
            ax.plot(track11[:, 0], track11[:, 1], track11[:, 2])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$y$", labelpad=10)
ax.set_zlabel(r"$z$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)


plt.show()
Figure 6e
import numpy as np
from scipy.integrate import odeint # type: ignore
import matplotlib.pyplot as plt

# Define fonts to prevent box display and increase font size
plt.rcParams['font.sans-serif'] = ['STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 25  # Default font size
plt.rcParams['axes.labelsize'] = 25  # Axis label font size
plt.rcParams['xtick.labelsize'] = 25  # X-axis tick label font size
plt.rcParams['ytick.labelsize'] = 25  # Y-axis tick label font size
plt.rcParams['legend.fontsize'] = 25  # Legend font size

x = 1  


def function(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
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
    return [dxdt, dydt, dzdt, dwdt]  


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

t = np.arange(0, 50, 0.005) 

args = (80, 0.02, 50.38, 60, 0.06, 0.5, 0.08, 0.56, 1, 2, 0.8, 1, 0.03, 0.01, 0.61, 8, 0.02, 0.01, 0.02, 0.02, 0.01, 1)


for j in np.arange(0.1, 1, 0.2):  
    for k in np.arange(0.1, 1, 0.2):  
        for w in np.arange(0.1, 1, 0.2): 
          
            track11 = odeint(function, (x, j, k, w), t, args)
           
            ax.plot(track11[:, 1], track11[:, 2], track11[:, 3])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$y$", labelpad=10)
ax.set_ylabel(r"$z$", labelpad=10)
ax.set_zlabel(r"$w$", labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


plt.show()
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1, step=0.2), [])
plt.yticks(np.arange(0, 1, step=0.2), [])

plt.text(0.8, 0.1, s="x", transform=ax1.transAxes, fontsize=25)
plt.text(0.1, 0.8, s="z", transform=ax1.transAxes, fontsize=25)  # Update label to y
plt.text(0.46, 0.02, s="0", transform=ax.transAxes, fontsize=25)

plt.show()
Figure 6f
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

y_fixed = 1  


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
    for z in np.arange(0.1, 1, 0.2):  
        for w in np.arange(0.1, 1, 0.2):  
            
            track11 = odeint(fuction, (x, y_fixed, z, w), t, args) 
            
            ax.plot(track11[:, 0], track11[:, 2], track11[:, 3])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$z$", labelpad=10)
ax.set_zlabel(r"$w$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)


plt.show()
Figure 6g
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
Figure 6h
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

w = 1  



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
        for z in np.arange(0.1, 1, 0.2):  
           
            track11 = odeint(fuction, (x, y, z, w), t, args) 
            
            ax.plot(track11[:, 0], track11[:, 1], track11[:, 2])


ax.view_init(elev=25, azim=-45) 
ax.set_facecolor('w')
ax.set_xlabel(r"$x$", labelpad=10)
ax.set_ylabel(r"$y$", labelpad=10)
ax.set_zlabel(r"$z$", labelpad=10)
ax.set_xlim3d(xmin=0, xmax=1)
ax.set_ylim3d(ymin=0, ymax=1)
ax.set_zlim3d(zmin=0, zmax=1)


plt.show()
Figure 7a-c
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


plt.rcParams['font.sans-serif'] = ['Arial']  
plt.rcParams['axes.unicode_minus'] = False


def function(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
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


Sg = 80
Cq2 = 0.02
Tg = 50.38
Lg = 60
Rt2 = 0.06
alpha = 0.5
Rt1 = 0.08
Cp2 = 0.56  
Rv = 2
Rp2 = 0.8
Υt = 1
Cq1 = 0.03
Cq3 = 0.01
Cp1 = 0.61
L = 8
Dt = 0.02
Ft = 0.01
Υp = 0.02
Dp = 0.02
Fp = 0.01
Rp1 = 1

t = np.linspace(0, 5, 100)


h_values = [0.5, 1, 1.5]
initial_conditions = [0.1, 0.1, 0.1, 0.1]


colors = ['#1f77b4',  
          '#17becf', 
          '#2ca02c',  
          '#9467bd']  


for h in h_values:
    plt.figure(figsize=(10, 6))
    
   
    solution = odeint(function, initial_conditions, t, args=(Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1))
    
  
    x, y, z, k = solution.T
    
   
    plt.plot(t, x, label='x', color=colors[0], marker='D', linestyle='-', linewidth=1.5)  
    plt.plot(t, y, label='y', color=colors[1], marker='*', linestyle='--', linewidth=1.5)
    plt.plot(t, z, label='z', color=colors[2], marker='s', linestyle=':', linewidth=1.5) 
    plt.plot(t, k, label='k', color=colors[3], marker='o', linestyle='-.', linewidth=1.5)  
    
   
    plt.title('', fontsize=25, fontstyle='italic')  
    plt.xlabel('Time', fontsize=25, fontstyle='italic')
    plt.ylabel('Probability', fontsize=25, fontstyle='italic')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

plt.figure(figsize=(10, 6))

for h in h_values:
   
    solution = odeint(function, initial_conditions, t, args=(Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1))
    
    
    x, y, z, k = solution.T
    
    
    plt.plot(t, x, label=f'x (h={h})', color=colors[0], marker='D', linestyle='-', linewidth=1.5) 
    plt.plot(t, y, label=f'y (h={h})', color=colors[1], marker='*', linestyle='--', linewidth=1.5)  
    plt.plot(t, z, label=f'z (h={h})', color=colors[2], marker='s', linestyle=':', linewidth=1.5)  
    plt.plot(t, k, label=f'k (h={h})', color=colors[3], marker='o', linestyle='-.', linewidth=1.5)  


plt.title('', fontsize=25, fontstyle='italic')
plt.xlabel('Time', fontsize=25, fontstyle='italic')
plt.ylabel('Probability', fontsize=25, fontstyle='italic')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=20, loc='upper right', frameon=True)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
Figure 7d-f
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


plt.rcParams['font.sans-serif'] = ['Arial']  
plt.rcParams['axes.unicode_minus'] = False


def function(Fx, t, Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1):
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


Sg = 80
Cq2 = 0.02
Tg = 50.38
Lg = 60
Rt2 = 0.06
alpha = 0.5
Rt1 = 0.08
Cp2 = 0.56  
Rv = 2
Rp2 = 0.8
Υt = 1
Cq1 = 0.03
Cq3 = 0.01
Cp1 = 0.61
L = 8
Dt = 0.02
Ft = 0.01
Υp = 0.02
Dp = 0.02
Fp = 0.01
Rp1 = 1


t = np.linspace(0, 5, 100)


h_values = [0.5, 1, 1.5]
initial_conditions = [0.1, 0.1, 0.1, 0.1]


colors = ['#1f77b4', 
          '#17becf',  
          '#2ca02c',  
          '#9467bd']


for h in h_values:
    plt.figure(figsize=(10, 6))
    
  
    solution = odeint(function, initial_conditions, t, args=(Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1))
    
    
    x, y, z, k = solution.T
    
   
    plt.plot(t, x, label='x', color=colors[0], marker='D', linestyle='-', linewidth=1.5)  
    plt.plot(t, y, label='y', color=colors[1], marker='*', linestyle='--', linewidth=1.5)  
    plt.plot(t, z, label='z', color=colors[2], marker='s', linestyle=':', linewidth=1.5)  
    plt.plot(t, k, label='k', color=colors[3], marker='o', linestyle='-.', linewidth=1.5)  
    
   
    plt.title('', fontsize=25, fontstyle='italic')  
    plt.xlabel('Time', fontsize=25, fontstyle='italic')
    plt.ylabel('Probability', fontsize=25, fontstyle='italic')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


plt.figure(figsize=(10, 6))

for h in h_values:
   
    solution = odeint(function, initial_conditions, t, args=(Sg, Cq2, Tg, Lg, Rt2, alpha, Rt1, Cp2, h, Rv, Rp2, Υt, Cq1, Cq3, Cp1, L, Dt, Ft, Υp, Dp, Fp, Rp1))
    
   
    x, y, z, k = solution.T
    
   
    plt.plot(t, x, label=f'x (h={h})', color=colors[0], marker='D', linestyle='-', linewidth=1.5)  
    plt.plot(t, y, label=f'y (h={h})', color=colors[1], marker='*', linestyle='--', linewidth=1.5)  
    plt.plot(t, z, label=f'z (h={h})', color=colors[2], marker='s', linestyle=':', linewidth=1.5)  
    plt.plot(t, k, label=f'k (h={h})', color=colors[3], marker='o', linestyle='-.', linewidth=1.5)  


plt.title('', fontsize=25, fontstyle='italic')
plt.xlabel('Time', fontsize=25, fontstyle='italic')
plt.ylabel('Probability', fontsize=25, fontstyle='italic')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=20, loc='upper right', frameon=True)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
