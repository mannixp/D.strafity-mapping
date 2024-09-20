from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root_scalar

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

"""
See Pope 1991, Mapping Closures...
This code implements a 1d version of the mapping closure for buoyancy.
The appropriate forcing is determined by specifying a desired PDF (here specified in terms of 'layers')

One of the issues discovered is that the forcing D^(1) which produces the multimodal pdf f_Y corresponidng to sratifications
has a discontinuity and indeed a discontinuous derivative.
"""


def Derivatives(N, dx):

  D = np.zeros((N,N))
  for i in range(N):
    if i < N-1:
      D[i,i+1] = 1
    if i > 0:
      D[i,i-1] = -1

  L = np.zeros((N,N))
  for i in range(N):

    L[i,i] = -2.
    if i ==N-1:
      L[i,i-1] = 2
      #L[i,i-1] = 1
    elif i == 0:
      L[i,i+1] = 2
      #L[i,i+1] = 1
    else:
      L[i,i-1] = 1
      L[i,i+1] = 1

  return D/(2*dx), L/(dx**2)


def operator(z):
    """Build the linear operator for the mapping equation."""

    N = len(z)
    dz = z[1]-z[0]
    grad, laplace = Derivatives(N, dz)

    return -np.diag(z)@grad + laplace


def F_Z(z):
    """Cumulative Gaussian distribution."""
    return (1+erf(z/np.sqrt(2)))/2


def f_Z(z):
    """Gaussian distribution."""
    return np.exp(-z**2/2)/np.sqrt(2*np.pi)


def make_F_Y(loc, std, amp):
    """
    Make the cumulative distribution function of Y in terms of layers
    Inputs:
        loc - mean of CDF/location of the layer

        std - standard deviation/width of the layer

        amp - amplitude/relative size of the layer compared to the others

    Returns:
        F_Y = sum_n F_Y^n - a sum of the CDFs for every layer
    """
    def F_Y(y):
        Y = 0*y
        for n in range(len(loc)):
            Y += amp[n]*(1 + erf( (y-loc[n])/(np.sqrt(2)*std[n]) ))/2
        return Y
    return F_Y


def make_f_Y(loc, std, amp):
    """
    Make the probability distribution function of Y in terms of layers
    Inputs:
        loc - mean of CDF/location of the layer

        std - standard deviation/width of the layer

        amp - amplitude/relative size of the layer compared to the others

    Returns:
        f_Y = sum_n f_Y^n - a sum of the PDFs for every layer
    """
    def f_Y(y):
        Y = 0*y
        for n in range(len(loc)):
            Y += ( amp[n]/np.sqrt(2*np.pi*std[n]**2) )*np.exp( -((y-loc[n])**2)/(2*std[n]**2) )
        return Y
    return f_Y


def make_map(F_Y, z):
    """Calculate the inverse of F_Y"""
    Y_ = []
    for z_i in z:
        func = lambda x: F_Y(x) - F_Z(z_i)
        sol = root_scalar(func, bracket=[-10*max(z), 10*max(z)], method='brentq')
        Y_.append(sol.root)

    return np.asarray(Y_)


def forcing(F_B, z):

    L = operator(z)

    # Solve F_Y( Y_ (z, t), t) = F_Z(z), where Y_ is the mapping
    Y_ = make_map(F_B, z)

    # Determine the forcing corresponding to the 'solution' Y_
    D1 = -L @ Y_

    return D1, L, Y_


def time_step(X_0, D1, L, Nt, T_end, J=lambda t: 1):

    Nz = L.shape[0]
    II = np.eye(Nz)
    T, Δt = np.linspace(0, T_end, Nt, retstep=True)

    LHS = lambda t: II/Δt - J(t)*L/2
    RHS = lambda t: II/Δt + J(t)*L/2

    X = np.zeros((Nt, Nz))
    X[0, :] = X_0
    for n, t in enumerate(T[:-1]):
        X[n+1, :] = np.linalg.solve(LHS(t), RHS(t)@X[n, :] + D1)

        # if n % 100 == 0:
        #     z, dz = np.linspace(-Lz/2, Lz/2, N, retstep=True)
        #     D = Derivatives(N, dz)[0]
        #     dY_dz = D @ X[n, :]
        #     f_ = f_Z(z)/dY_dz
        #     plt.plot(X[n, :], f_)
        #     plt.show()
        #     print('t =  %d (s) \n' % t)

    return X, T


def animation(X, N, Lz):
    """Generate an animation of the mapping equation. """

    z, dz = np.linspace(-Lz/2, Lz/2, N, retstep=True)
    D = Derivatives(N, dz)[0]
    
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), layout='constrained')

    ax1.set_xlabel(r'$y$', fontsize=20)
    ax1.set_ylabel(r'$f_Y$', fontsize=20)
    ax1.set_xlim([0, 1])
    #ax2.set_ylim([-.1, .1])
    ax1.grid(True)

    ax2.set_xlabel(r'$y$', fontsize=20)
    ax2.set_ylabel(r'$F_Y$', fontsize=20)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True)

    def animation_function(i):
        
        dY_dz = D @ X[i, :]
        f_ = f_Z(z)/dY_dz
        F = cumulative_trapezoid(y=f_, x=X[i, :], initial=0)

        for line in ax1.get_lines():
            line.remove()
        ax1.set_ylim([0,np.max(f_)])
        line1 = ax1.plot(X[i, :], f_, 'b', linewidth=2)

        for line in ax2.get_lines():
            line.remove()
        line2 = ax2.plot(X[i, :], F, 'k-', linewidth=2)

        return line1, line2

    anim = FuncAnimation(fig=fig, func=animation_function, frames=np.arange(0, len(X), 10))
    anim.save('Part2_Mapping_SIM.mp4', fps=10)
    
    return None


if __name__ == "__main__":
    # %%
    # 1) Set up a domain
    N = 256

    # z is the argument to the Gaussian distribution
    Lz = 10
    z = np.linspace(-Lz/2, Lz/2, N)

    # %%
    # 2) Generate a PDF corresponding to a two layer stratification

    # Location of 'interface(s)'
    loc = [0.25, 0.75]  # Mean/height of the layer
    std = [0.075, 0.075]  # Standard deviation of the layer
    amp = [1, 1]  # Changes the maximum and minimum buoyancy of the layer

    # Enforce normalisation
    SUM = np.sum(amp)
    amp = [i/SUM for i in amp]

    y = np.linspace(0, 1, N)  # Argument for the PDF F_Y

    F_Y = make_F_Y(loc, std, amp)
    f_Y = make_f_Y(loc, std, amp)

    plt.plot(y, F_Y(y))
    plt.ylabel(r'$F_Y$', fontsize=20)
    plt.xlabel(r'$y$', fontsize=20)
    plt.show()

    plt.plot(y, f_Y(y))
    plt.fill_between(y, f_Y(y))
    plt.ylabel(r'$f_Y$', fontsize=20)
    plt.xlabel(r'$y$', fontsize=20)
    plt.show()

    # %%
    # 3) Generate the forcing and the map Y_ required to produce this pdf
    D1, L, Y_ = forcing(F_Y, z)

    # Check the mapped pdf matches
    dz = z[1]-z[0]
    D = Derivatives(N, dz)[0]

    dY_dz = D @ Y_
    f_ = f_Z(z)/dY_dz

    fig = plt.figure(figsize=(6, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(Y_, f_, label=r'True $f_Y$')
    ax1.plot(y, f_Y(y), label=r'Mapped $f_Y$')
    ax1.fill_between(Y_, f_)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.1*np.max(f_)])
    ax1.set_xlabel(r'$y$', fontsize=20)
    ax1.set_ylabel(r'$f_Y$', fontsize=20)
    ax1.legend(loc=2)

    ax2.plot(Y_[1:-1], D1[1:-1], '-r', linewidth=2, label=r'$\mathbb{D}^1$')
    ax2.plot(Y_, 0*Y_, '-k')
    ax2.set_xlim([0, 1])
    ax2.set_ylabel(r'$\mathbb{D}^{(1)}$', fontsize=20)
    ax2.legend()

    plt.show()

    # %%
    # 4) Solve the time evolution of the mapping equation
    # from a Gaussian initial condition

    X_0 = np.linspace(0, 1, len(z))
    Nt = 2*(10**3)
    T_end = 1
    X, t = time_step(X_0, D1, L, Nt, T_end, J=lambda t: 1)

    # %%
    # 5) Produce the animation of this time evolution
    animation(X, N, Lz)

    # %%
    # To generate a three layered stratification repeat the previous
    # steps by changing the following parameters.

    # loc = [0.2, 0.5, 0.75]  # Mean/height of the layer
    # std = [0.06, 0.06, 0.06]  # Standard deviation of the layer
    # amp = [4, 2, 0.5]  # Changes the maximum and minimum buoyancy of the layer


