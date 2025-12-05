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