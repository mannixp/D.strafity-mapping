"""
Periodic box using an adapted version of Lundgren's forcing:

epsilon * u / <|u|^2>

where epsilon is the (eventual) dissipation, giving scales

L* = L
U* = (epsilon * L)^(1/3)
t* = L^(2/3) epsilon^(-1/3)

Ri_B = N^2 * L^(4/3) / epsilon^(2/3)
Re   = epsilon^(1/3) * L^(4/3) / nu

where in A. Maffioli et al. JFM (2016) these are written as

Fr^2 = 1/Ri_B
Re_B = Re Fr^2

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python simulate.py
"""

# Prevent multi-threading upon initialising mpi4py
from mpi4py import MPI
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Parameters
Lv = 1
Lh = 2.5*Lv
Nv = 64
Nh = 128

Re = 2.8*(10**3)
Fr = 1.6/(10**2)
Ri_B = 1/(Fr**2)
Re_B = Re*(Fr**2)
Pr = 1

# Write to a file
Params = {"Re":Re,"Fr":Fr,"Re_B":Re_B,"Pr":Pr,"Ri_B":Ri_B,"Nh":Nh,"Nv":Nv, "Lh":Lh,"Lv":Lv}
with open("Parameters.txt", "w") as file:
    for key, value in Params.items():
        file.write(f"{key}: {str(value)}\n")


dealias = 3/2
stop_sim_time = .1
timestepper = d3.RK222
max_timestep = 1e-04
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=Nh, bounds=(-Lh/2, Lh/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nh, bounds=(-Lh/2, Lh/2), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nv, bounds=(-Lv/2, Lv/2), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
b = dist.Field(name='b', bases=(xbasis, ybasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b = dist.Field(name='tau_b')
tau_u = dist.VectorField(coords, name='tau_u')

# Substitutions
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

curl = lambda A: d3.Curl(A)
ω    = curl(u)

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b, tau_u], namespace=locals())
problem.add_equation("dt(b) - lap(b)/(Re*Pr)                + u@ez + tau_b = -(u@grad(b))")
problem.add_equation("dt(u) - lap(u)/(Re   ) + grad(p) - Ri_B*b*ez + tau_u = cross(u,ω) + u/integ(u@u) ")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")
problem.add_equation("integ(b) = 0")
problem.add_equation("integ(u) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-0) # Random noise
u.fill_random('g', seed=98, distribution='normal', scale=1e-0) # Random noise
u = u - d3.Average(u)
b = b - d3.Average(b)

#write, initial_timestep = solver.load_state("checkpoints/checkpoints_s1.h5")

# Analysis
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=1)
checkpoints.add_tasks(solver.state)

# Snapshots
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=.1)

snapshots.add_task(b   , layout='g', name='b', scales=2)
snapshots.add_task(u@ex, layout='g', name='u', scales=2)
snapshots.add_task(u@ey, layout='g', name='v', scales=2)
snapshots.add_task(u@ez, layout='g', name='w', scales=2)

snapshots.add_task(b   , layout='c', name='b_k', scales=1)
snapshots.add_task(u@ex, layout='c', name='u_k', scales=1)
snapshots.add_task(u@ey, layout='c', name='v_k', scales=1)
snapshots.add_task(u@ez, layout='c', name='w_k', scales=1)

# snapshots.add_task(d3.grad(b)   , name='grad_b',scales=1)
# snapshots.add_task(d3.grad(u@ez), name='grad_w',scales=1)
# snapshots.add_task(d3.grad(p)   , name='grad_p',scales=1)

# Time Series and spectra
scalar = solver.evaluator.add_file_handler('scalar_data', iter=50)
scalar.add_task(d3.Integrate(u@u ),  layout='g', name='Eu(t)')
scalar.add_task(d3.Integrate(b**2),  layout='g', name='Eb(t)')
scalar.add_task(d3.Integrate(d3.grad(u@ez)@d3.grad(u@ez) + d3.grad(u@ey)@d3.grad(u@ey) + d3.grad(u@ex)@d3.grad(u@ex))/Re, layout='g', name='dU^2(t)/Re')
scalar.add_task(d3.Integrate(d3.grad(b)@d3.grad(b))/(Re*Pr), layout='g', name='dB^2(t)/(Re*Pr)')

Z = dist.Field(name='Z', bases=zbasis); Z['g'] = z[:];
scalar.add_task(d3.Integrate((u@ez)*b),  layout='g', name='<wB>')
scalar.add_task(d3.Integrate(b)       ,  layout='g', name='<B>' )
scalar.add_task(d3.Integrate(Z*b)     ,  layout='g', name='<zB>')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.35, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(d3.Integrate((u@ez)*b), name='<wB>')
flow.add_property(d3.Integrate(d3.grad(u@ez)@d3.grad(u@ez) + d3.grad(u@ey)@d3.grad(u@ey) + d3.grad(u@ex)@d3.grad(u@ex))/Re, name='dU^2/Re')
flow.add_property(d3.Integrate(d3.grad(b)@d3.grad(b))/(Re*Pr), name='dB^2/(Re*Pr)')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = max_timestep 
        #timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:

            wB_avg = flow.grid_average('<wB>')
            dU_avg = flow.grid_average('dU^2/Re')
            dB_avg = flow.grid_average('dB^2/(Re*Pr)')
            
            logger.info('Iteration=%i, Time=%e, dt=%e'%(solver.iteration, solver.sim_time, timestep))
            logger.info('<wB>=%f, <dU^2>/Re =%f, <dB^2>/(Re*Pr) =%f'%(wB_avg,dU_avg,dB_avg))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
