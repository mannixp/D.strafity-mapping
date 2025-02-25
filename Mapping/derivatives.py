"""
Code to define and test the gradient & Laplacian operators
"""
import numpy as np
from scipy import sparse
from scipy.integrate import trapezoid as trapz


def grad(x):
    """Gradient operator ∂/∂x"""
    N = len(x)
    dx = x[1] - x[0]
    D = np.zeros((N, N))
    for i in range(N):
        if i < N-1:
            D[i, i+1] = 1
        if i > 0:
            D[i, i-1] = -1
    return D/(2*dx)


def laplacian(x, bcs='Neumann'):
    """Laplacian ∂^2/∂x^2"""

    N = len(x)
    dx = x[1] - x[0]
    L = np.zeros((N,N))
    for i in range(N):

        L[i, i] = -2.
        if i == N-1:
            if bcs == 'Dirichlet':
                L[i, i-1] = 1
            elif bcs == 'Neumann':
                L[i, i-1] = 2
        elif i == 0:
            if bcs == 'Dirichlet':
                L[i, i+1] = 1
            elif bcs == 'Neumann':
                L[i, i+1] = 2
        else:
            L[i, i-1] = 1
            L[i, i+1] = 1

    return L/(dx**2)


def sparse_grad(x):
    """Gradient operator ∂/∂x"""
    N = len(x)
    dx = x[1] - x[0]
    Akp1 = np.ones(N-1)
    Akm1 = -np.ones(N-1)
    return sparse.diags([Akm1, Akp1], [-1, 1])/(2*dx)


def sparse_laplacian(x, bcs='Dirichlet'):
    """Laplacian ∂^2/∂x^2"""

    N = len(x)
    dx = x[1] - x[0]
    Akp1 = np.ones(N-1)
    Ak0 = -2*np.ones(N)
    Akm1 = np.ones(N-1)

    if bcs == 'Neumann':
        Akp1[0] = 2
        Akm1[-1] = 2
    elif bcs == 'Dirichlet':
        Akp1[0] = 0
        Ak0[0] = 0
        Ak0[-1] = 0
        Akm1[-1] = 0

    return sparse.diags([Akm1, Ak0, Akp1], [-1, 0, 1])/(dx**2)


def test_derivatives():
    """Check the derivatives are correctly implemented"""

    x = np.linspace(0, 2*np.pi, 256)
    tol = 1e-07

    D = grad(x)
    error_D = (np.cos(x)[1:-1] - D.dot(np.sin(x))[1:-1])**2
    assert trapz(y=error_D, x=x[1:-1]) < tol

    L = laplacian(x)
    error_L = (-np.sin(x)[1:-1] - L.dot(np.sin(x))[1:-1])**2
    assert trapz(y=error_L, x=x[1:-1]) < tol

    D = sparse_grad(x)
    error_D = (np.cos(x)[1:-1] - D.dot(np.sin(x))[1:-1])**2
    assert trapz(y=error_D, x=x[1:-1]) < tol

    L = sparse_laplacian(x)
    error_L = (-np.sin(x)[1:-1] - L.dot(np.sin(x))[1:-1])**2
    assert trapz(y=error_L, x=x[1:-1]) < tol

    return None


if __name__ == "__main__":

    test_derivatives()
