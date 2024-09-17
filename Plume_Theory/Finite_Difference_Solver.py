from   scipy import sparse
import scipy.sparse.linalg as sla
import numpy as np
import time
import matplotlib.pyplot as plt


# Base class for finite-differences
class FD_Solve(object):

    """
    A simple base class for finite difference simulations
    """
    def __init__(self,N,domain,dt=0.1,endTime=1):

        # Temporal params
        self.dt = dt
        self.endTime = endTime
        self.Nt = int(self.endTime/self.dt)
        self.t  = 0.

        # Domain & grid-spacings, use strings to handle n-dimensions
        self.N  = N
        self.V  = 1.0
        self.dV = 1.0

        #Construct basis and dx and volume for each of the directions input
        for basis in domain:

            bases = np.linspace(basis['interval'][0],basis['interval'][1],self.N)
            setattr(self,basis['name'],bases)

            d_bases = abs(bases[1] - bases[0])
            setattr(self,'d'+basis['name'],d_bases)

            self.dV *= d_bases;
            self.V  *= abs(bases[-1] - bases[0])

    # Differentiation matrices
    def Grad(self):

        # Gradient - must be multiplied by 1/dx_i

        N = self.N;

        Akp1 =      np.ones(N-1);
        Akm1 = -1.0*np.ones(N-1);

        return 0.5*sparse.diags( [Akm1,Akp1], [-1,1] );

    def Laplacian(self, bcs = 'Dirichlet'):

        # Laplacian operator with:
        # bcs = 'Dirichlet' boundary conditions
        # bcs = 'Neumann'   boundary conditions
        # - must be multiplied by 1/dx^2_i
        N = self.N;

        Akp1 =      np.ones(N-1);
        Ak0  = -2.0*np.ones(N  );
        Akm1 =      np.ones(N-1);

        # Boundary Condiitions
        if bcs == 'Neumann':
            Akp1[0 ] = 2.0; # Use f_1     = f_-1    @ f_0
            Akm1[-1] = 2.0; # Use f_{N+1} = f_{N-1} @ f_N
        elif bcs == 'Dirichlet':
            Akp1[0]  = 0.
            Ak0[0]   = 0.; Ak0[-1] =0.
            Akm1[-1] = 0.

        return sparse.diags( [Akm1,Ak0,Akp1], [-1,0,1] )

    # Integration routines
    def Volume_Integrate(self):

        """
        Use the grid to volume integrate
        """

        raise NotImplementedError

    def Normalise(self):

        """
        Rescale the probability distribution  s.t.
        int_V P dV = 1

        returns scaled P
        """
        self.P *= 1./(self.dV*np.sum(self.P));
        return None;

    # Time-stepping routines
    def Time_Integrate(self,θ=0.5):

        """
        Time integrates the Fokker-Planck equation

        dP/dt = -L*P + F(P,t) on Ω

        This can be re-written as

        [ P^{n+1} - P^n ]/Δt = -L*[ (1-θ)*P^{n+1} + θ*P^n ]  +  F(P^n,t^n)

        A = I/Δt + (1-θ)*L
        B = I/Δt +     θ*L

        using:
        θ = 1   Forward Euler
        θ = 1/2 Crank-Nicolson scheme
        θ = 0   Backwards Euler
        """

        # Grab Operators
        L = self.Lin_operator()
        F = lambda P,t: self.NLin_operator(P,t)
        # try:
        #   F = lambda P,t: self.NLin_operator(P,t)
        # except RuntimeError as error:
        #   print(error)
        # else:
        #   F = lambda P,t: 0*P

        # Build matrices
        Δt = self.dt
        I  = sparse.eye(L.shape[0],format="csr")
        B  = sparse.csr_matrix(I/Δt - θ*L)
        if θ == 1:
          A = I/Δt
        else:
          A  = sparse.csr_matrix(I/Δt + (1-θ)*L)

        # time step
        st = time.time()
        for t_i in range(self.Nt):

          b = B.dot(self.P) + F(self.P,self.t)
          self.P = sla.spsolve(A,b)
          self.t+=self.dt

        print('Elapsed time (s) = %e'%(time.time()-st))
        print('Conserved mass: int_v P dv =',np.sum(self.P)*self.dV)

        return None;

# Solver class for 1D Ornstein-Ulhenbeck
class OrnsteinUlhenbeck(FD_Solve):

    '''
    Solver class derived from FD solve which solves the Fokker-Planck equation
    underlying the Ornstein-Ulhenbeck process and compares it with the exact solution

    Inputs
    -a,gamma parameters (float)
    -N grid-size (int)
    -domain (dictionary) e.g.  {'name':'x','interval':(-1,1)}

    Methods:
    -init
    -operator
    -Histogram
    '''

    def __init__(self,a,μ,gamma,N,domain,endTime=1,dt=0.01):

        # Initialise the base clas
        super().__init__(N,domain,dt,endTime)

        self.a = a
        self.μ = μ
        self.gamma = gamma;

        # Probability Field
        f = lambda x: np.sqrt(self.a/(2.*np.pi*self.gamma))*np.exp(-self.a*((x-self.μ)**2)/(2.*self.gamma) )
        print('ic mass = ',np.sum(f(self.x)*self.dx))

        self.P = f(self.x-0.2)

        return None;

    def Lin_operator(self):

        """
        Construct the linear operator

        L = -γ*∂^2P/∂x^2

        """
        # (1) Create Laplacian [1/(dx)^2]*L
        Lap   = self.Laplacian();
        Lap_x = pow(self.dx,-2)*Lap;

        return sparse.csr_matrix( -self.gamma*Lap_x)

    def NLin_operator(self,P,t):

        """
        Evaluate the nonlinear operator

        F(P(x,t),t) = −∂/∂x[a(μ−x)P]

        """
        # (1) Differentiation matrix
        I  = sparse.eye(self.N);
        D  = self.Grad()
        Dx = pow(self.dx,-1)*D
        μ_x= self.a*( self.μ*I - sparse.diags(self.x))

        return -Dx@( μ_x@P  )

    def Histogram(self):

        f = lambda x: np.sqrt(self.a/(2.*np.pi*self.gamma))*np.exp(-self.a*((x-self.μ)**2)/(2.*self.gamma))
        p = f(self.x)
        p *= 1./(self.dx*np.sum(p))
        plt.plot(self.x,p     ,'k-.',label=r'exact')
        plt.plot(self.x,self.P,'k'  ,label=r'num');
        plt.ylabel(r'$f_x(\tilde{x})$',fontsize=20)
        plt.xlabel(r'$\tilde{x}$',fontsize=20)
        plt.xlim([self.x.min(),self.x.max()])
        plt.legend()
        plt.show()

        return None;

# Solver class for duffing equation
class DuffingOscillator(FD_Solve):

    '''
    Solver class derived from FD solve which solves the Foker-planck equation underlying the Duffing equation forced by a Wiener process

    Inputs
    -N grid-size (int)
    -domain (dictionary) e.g.  {'name':'x','interval':(-1,1)}

    Methods:
    -init
    -operator
    -Histogram
    '''

    def __init__(self,N,domain,dt=0.01,endTime=10.):


        # Initialise the base clas
        super().__init__(N,domain,dt,endTime)

        # Set the system parameters
        self.ζ = 0.2
        self.ω = 1.0
        self.γ = 0.1
        self.D = 0.4

        # build the drift & Diffusion terms
        self.Drift_μ()
        #self.Diffusion_D()

        # Probability Field
        f = lambda x: (1./np.sqrt(2.*np.pi))*np.exp(-x**2 /2.)
        self.P = np.kron( f(self.x), f(self.y) )
        print('Initial mass =',np.sum(self.P*self.dV))

    def Drift_μ(self):

      # Create the nonlinear part
      II = sparse.eye(self.N);
      D  = self.Grad();
      self.Dx = pow(self.dx,-1)*sparse.kron( D, II);
      self.Dy = pow(self.dy,-1)*sparse.kron( II, D);

      # (2) Create flat arrays of each of these vectors using Kronecker product
      I = np.ones(self.N)
      X = np.kron(self.x,I)
      Y = np.kron(I,self.y)

      # (3) Create the vector field
      self.F1 = Y;
      self.F2 = -2.*self.ζ*self.ω*Y + (self.ω**2)*X - (self.ω**2)*self.γ*(X**3)

      return None

    # def Diffusion_D(self):

    #     return None;

    def Lin_operator(self):

      """
      Build the lineat diffusion term

      L = -D*∂^2P/∂y2

      """

      # (1) Create a 2D mass matrix by using the kronecker product [1/(dx*dy)^2]*[ (L x L) ]
      II     = sparse.eye(self.N);
      Lap    = self.Laplacian();
      Lap_yy = self.D*pow(self.dy,-2)*sparse.kron(II,Lap)

      return -sparse.csr_matrix(Lap_yy)

    def NLin_operator(self,P,t):

      """
      Build the nonlinear drift term

      F(P,t) = −∂/∂x[yP]−∂/∂x[(−2ζωy+ω^2x−ω^2γx^3)P]

      """

      return -self.Dx@(self.F1*P) - self.Dy@(self.F2*P)

    def Histogram(self):

        P = self.P;
        N = self.N;

        # Add plotting here
        # P is structured as ~ (X kron Y) kron Z
        # Reconstruct this as a 3D matrix
        W = P.reshape( (N,N) );

        fig = plt.figure()
        X, Y = np.meshgrid(self.x, self.y)
        #cf = plt.pcolormesh(X,Y,H.T,cmap='Greys',vmax=np.max(H)) # Normalise this + show the colour bar ,norm="log"
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, W.T, cmap='Greys',linewidth=0, antialiased=False)

        fig.colorbar(surf)
        plt.show()

        return None

# x_basis = {'name':'x','interval':(-1,1)}
# FP = OrnsteinUlhenbeck(a=1,μ=-0.1,gamma=0.01,N=128,domain=[x_basis],dt=0.01,endTime=10)
# FP.Time_Integrate(θ = 1.0)
# FP.Histogram()

# x_basis = {'name':'x','interval':(-6,6)}
# y_basis = {'name':'y','interval':(-6,6)}
# FP = DuffingOscillator(N=96,domain=[x_basis,y_basis],dt=0.00125,endTime=5.);
# FP.Time_Integrate(θ = 0.5)
# FP.Histogram()