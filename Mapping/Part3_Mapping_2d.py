from scipy.special import erf, erfinv
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from numpy.linalg import inv

from Mapping.helper import *

"""
See Pope 1991, Mapping Closures...
This code implements a 2d version of the mapping closure for buoyancy and vertical velocity.
"""

# Domain
lim1 = [-3,3]; N1 = 32; d1 = (lim1[1]-lim1[0])/N1;
lim2 = [-3,3]; N2 = 32; d2 = (lim2[1]-lim2[0])/N2;

def unflatten(u):
    # Convert vector to array
    return np.reshape(u, (N1+2, N2+2))

# Cell centred grid
z1 = np.linspace(lim1[0]-d1/2, lim1[1]+d1/2, N1+2)
z2 = np.linspace(lim2[0]-d2/2, lim2[1]+d2/2, N2+2)

# 2d grid for gaussian random variables (buoyancy Y1(z1) and vertical velocity Y2(z2))
z1_2d, z2_2d = np.meshgrid(z1, z2, indexing='ij')

I_1d = np.eye(N1+2)
D1_1d = (np.roll(I_1d, 1, axis=1) - np.roll(I_1d, -1, axis=1))/2/d1
DD_1d = (np.roll(I_1d, 1, axis=1) -2*I_1d + np.roll(I_1d, -1, axis=1))/d1**2

I_2d = shift_operator(0,0,N1+2,N2+2)
S1_2d = shift_operator(1,0,N1+2,N2+2)
S2_2d = shift_operator(0,1,N1+2,N2+2)
D1_2d = (S1_2d - inv(S1_2d))/2/d1
D2_2d = (S2_2d - inv(S2_2d))/2/d2
DD1_2d = (S1_2d - 2*I_2d + inv(S1_2d))/d1**2
DD2_2d = (S2_2d - 2*I_2d + inv(S2_2d))/d2**2

div = np.concatenate([(S1_2d - I_2d)/d1, (S2_2d - I_2d)/d2], axis=1)
grad = np.concatenate([D1_2d, D2_2d], axis=0)
laplace = div @ grad

south = mk_boundary( z2_2d < lim2[0], 0,  1, N1+2, N2+2)
north = mk_boundary( z2_2d > lim2[1], 0, -1, N1+2, N2+2)
west  = mk_boundary( z1_2d < lim1[0], 1,  0, N1+2, N2+2)
east  = mk_boundary( z1_2d > lim1[1],-1,  0, N1+2, N2+2)

def apply_bcs(Y):
    # Homogeneous Neumann BCs
    Y[south.id] = Y[south.id_]
    Y[north.id] = Y[north.id_]
    Y[west.id]  = Y[west.id_]
    Y[east.id]  = Y[east.id_]

# Time scales (cf. Taylor microscale)
t1 = 1
t2 = 1

L1_2d = (-np.diag(z1_2d.flatten()) @ D1_2d + DD1_2d)/t1
L2_2d = -np.diag(z1_2d.flatten()) @ D1_2d/t1 -np.diag(z2_2d.flatten()) @ D2_2d/t2 + DD1_2d / t1 + DD2_2d / t2

# Forcing due to buoyancy (see governing equations)
h_2d = z1_2d.flatten()

# Standard deviations for initial condition
s1 = 4
s2 = 2

Y1_0_2d = (erfinv(2*G(z1_2d) - 1)*s1*np.sqrt(2)).flatten()
Y2_0_2d = (erfinv(2*G(z2_2d) - 1)*s2*np.sqrt(2)).flatten()

apply_bcs(Y1_0_2d)
apply_bcs(Y2_0_2d)

dt = 0.001
nt = 2000

# I think Y1_0_2d is included in the second equation because of the buoyancy
# term in the equations
for i in range(nt):
    Y1_0_2d +=  L1_2d @ Y1_0_2d * dt
    Y2_0_2d += (L2_2d @ Y2_0_2d + Y1_0_2d)* dt

    apply_bcs(Y1_0_2d)
    apply_bcs(Y2_0_2d)

# Determine pdf f_Y(y1, y2)
f = g(z1_2d).flatten() * g(z2_2d).flatten() * (D1_2d @ Y1_0_2d)**(-1) * (D2_2d @ Y2_0_2d)**(-1)

plt.contourf(unflatten(Y1_0_2d), unflatten(Y2_0_2d), unflatten(f))
plt.colorbar()
plt.show()
