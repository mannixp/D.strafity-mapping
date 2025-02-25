"""
Plot data generated using main.py

python3 plot_homogeneous_3D.py

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })


def main(data_dir):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot writes
    with h5py.File(data_dir + "/snapshots/snapshots_s1.h5", mode='r') as file:
        b = file['tasks/b'][-1, :, 0, :]
        t = file['tasks/b'].dims[0][0][:]
        x = file['tasks/b'].dims[1][0][:]
        y = file['tasks/b'].dims[2][0][:]
        z = file['tasks/b'].dims[3][0][:]

        fig = plt.figure()
        plt.xlabel(r'x')
        plt.ylabel(r'z')
        plt.pcolormesh(x, z, b.T, cmap='RdBu')
        fig.savefig('homogeneous_RBC', dpi=100)
        fig.clear()

    plt.close(fig)

    return None


def video(data_dir):
    """Save plot of specified tasks for given range of analysis writes."""

    import matplotlib.animation as animation

    # Plot writes
    with h5py.File(data_dir + "/snapshots/snapshots_s1.h5", mode='r') as file:
        
        b = file['tasks/b'][:, :, 0, :]
        t = file['tasks/b'].dims[0][0][:]
        x = file['tasks/b'].dims[1][0][:]
        y = file['tasks/b'].dims[2][0][:]
        z = file['tasks/b'].dims[3][0][:]

        fig, ax = plt.subplots()
        ax.set_xlabel(r'x')
        ax.set_ylabel(r'z')
        quad = ax.pcolormesh(x, z, b[-1, :, :].T, cmap='RdBu')
        fig.colorbar(quad, ax=ax)
       
        # Animation function
        def update(frame):
            Z = b[frame, :, :].T
            quad.set_array(Z.ravel())  # Update the pcolormesh data
            return quad,

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=int(b.shape[0]), interval=10, blit=False)
        ani.save("animation.mp4", writer="ffmpeg", fps=5)

        plt.close(fig)

    return None



def average_over_discs_2D_v2(Φ, kx, ky, kz, N = 100):

    π = np.pi
    shape = Φ.shape

    # Create coordinate grids
    kx, ky = np.meshgrid(kx,ky,indexing="ij")

    # Compute radial distances
    r = np.sqrt(kx**2 + ky**2)

    # Create a 2D array
    kh = np.unique(r.flatten())
    Eh = np.zeros( (len(kh),len(kz)) )

    # Compute the E_l i.e. when |k| = r_l
    for k in range(shape[2]):
        
        radial_E = []
        radial_r = []
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                
                # Integrate over the circle assuming isotropic  dA = r dθ      
                r_l = r[i, j]
                E_l = (1/2)*Φ[i, j, k]*r_l*(2*π)
                radial_E.append(E_l)
                radial_r.append(r_l)
               
        E_k = np.asarray(radial_E)
        r_k = np.asarray(radial_r)

        # Sum all E_l associated with the same r_l to integrate over the spheres
        for i, r_i in enumerate(kh):
            indx = (r_k == r_i).nonzero()
            Eh[i, k] = np.mean( E_k[indx] )

    return Eh, kh


def average_over_discs_2D(arr, radius_bins):
    """
    Averages the array over discs of constant radius.

    Parameters:
    - arr: 2D numpy array with shape (x, y)
    - radius_bins: List or array of radius bin edges that define the discs

    Returns:
    - avg_per_disc: Array of average values for each disc
    - radii: Midpoint of each disc for reporting purposes
    """

    # Create a 2D grid of coordinates (x, y, z)
    x, y = np.indices(arr.shape)

    # Calculate the radial distance from the origin for each point
    r = np.sqrt(x**2 + y**2)

    # Initialize an array to store the averaged values for each disc
    avg_per_disc = []

    # Define shell radii (midpoint of each bin)
    radii = 0.5 * (radius_bins[1:] + radius_bins[:-1])

    # Loop through each shell
    for i in range(len(radius_bins) - 1):
        # Create a mask for the points within the current shell
        mask = (r >= radius_bins[i]) & (r < radius_bins[i + 1])
        
        # Extract values from the original array that lie within the shell
        values_in_disc = arr[mask]
        
        # Compute the average of the values in the shell
        if values_in_disc.size > 0:
            avg_per_disc.append(np.mean(values_in_disc))
        else:
            avg_per_disc.append(np.nan)  # In case the shell has no points

    return np.array(avg_per_disc), radii


def average_over_shells_3D_v2(Φ, kx, ky, kz):

    π = np.pi
    shape = Φ.shape

    # Create coordinate grids
    kx, ky, kz = np.meshgrid(kx,ky,kz,indexing="ij")

    # Compute radial distances
    r = np.sqrt(kx**2 + ky**2 + kz**2)

    # Compute the E_l i.e. when |k| = r_l
    radial_E = []
    radial_r = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):

                # Integrate over the shell assuming isotropic   dA = r^2 sin(θ) dθ d𝜑     
                r_l = r[i, j, k]
                E_l = (1/2)*Φ[i, j, k]*(r_l**2)*(4*π)
                radial_E.append(E_l)
                radial_r.append(r_l)
               
    E = np.asarray(radial_E)
    r = np.asarray(radial_r)

    # Sum all E_l associated with the same r_l to integrate over the spheres
    k   = np.unique(r)
    E_k = 0*k
    for i, r_i in enumerate(k):
        indx = (r == r_i).nonzero()
        E_k[i] = np.mean( E[indx] )

    return E_k, k


def average_over_shells_3D(arr, radius_bins):
    """
    Averages the array over spherical shells of constant radius.

    Parameters:
    - arr: 3D numpy array with shape (x, y, z)
    - radius_bins: List or array of radius bin edges that define the shells

    Returns:
    - avg_per_shell: Array of average values for each shell
    - radii: Midpoint of each shell for reporting purposes
    """

    # Create a 3D grid of coordinates (x, y, z)
    x, y, z = np.indices(arr.shape)

    # Calculate the radial distance from the origin for each point
    r = np.sqrt(x**2 + y**2 + z**2)

    # Initialize an array to store the averaged values for each shell
    avg_per_shell = []

    # Define shell radii (midpoint of each bin)
    radii = 0.5 * (radius_bins[1:] + radius_bins[:-1])

    # Loop through each shell
    for i in range(len(radius_bins) - 1):
        # Create a mask for the points within the current shell
        mask = (r >= radius_bins[i]) & (r < radius_bins[i + 1])
        
        # Extract values from the original array that lie within the shell
        values_in_shell = arr[mask]
        
        # Compute the average of the values in the shell
        if values_in_shell.size > 0:
            avg_per_shell.append(np.mean(values_in_shell))
        else:
            avg_per_shell.append(np.nan)  # In case the shell has no points

    return np.array(avg_per_shell), radii


def energy_spectrum(data_dir, N=100):
    """
    Compute the energy spectrum and the horizontally averaged spectra of the velocity field
    
    The energy spectrum see chapter 6 pope is defined as
    
    E(k,t) = < u_hat*(k,t) u_hat(k,t)>

    where 
    
    u(x) = sum u_hat e^ikx = sum (A_k + i*B_k)*(cos(k x) + i*sin(k x))

    thus

    E(k,t) = < u_hat*(k,t) u_hat(k,t)> = A_k**2 + B_k**2

    In Dedalus the amplitudes are stored as sine and cosine 

    u(x) = sum_k A_k cos(k x) - B_k sin(k x)
    
    therefore we just access A_k and B_k and square them.
    """

    f = h5py.File(data_dir + "/snapshots/snapshots_s1.h5", mode='r')
    
    # Ordering of coefficients is 
    # cos(0*x),sin(0*x), cos(1*x),sin(1*x), .... 
    # in each dimension
    b_k = f['tasks/b_k'][-1, :, :, :]

    u_k = f['tasks/u_k'][-1, :, :, :]
    v_k = f['tasks/v_k'][-1, :, :, :]
    w_k = f['tasks/w_k'][-1, :, :, :]
    
    t  = f['tasks/u_k'].dims[0][0][:]
    kx = f['tasks/u_k'].dims[1][0][::2, ::2, ::2]
    ky = f['tasks/u_k'].dims[2][0][::2, ::2, ::2]
    kz = f['tasks/u_k'].dims[3][0][::2, ::2, ::2]

    f.close()

    # 1) First square all the amplitudes
    R_ii = u_k**2 + v_k**2 + w_k**2
    B_k = b_k**2

    # 2) Convert to k=0,1,2,.. ordering
    R_ii = R_ii[::2, ::2, ::2] + R_ii[1::2, 1::2, 1::2]
    B_k  = B_k[::2, ::2, ::2]  +  B_k[1::2, 1::2, 1::2]
    
    # 3) Define the vector k = (kx,ky,kz)
    kx = np.unique(kx)
    ky = np.unique(ky)
    kz = np.unique(kz)

    # 4) Convert to the energy spectrum i.e. E(|k|) vs. |k|     
    Eu_1d, k = average_over_shells_3D_v2(R_ii,kx, ky, kz)
    Eb_1d, k = average_over_shells_3D_v2(B_k, kx, ky, kz)

    # 5) Convert to the 2D energy spectrum i.e. E(k_h,k_z) vs. k_h = |(k_x,k_y)|, k_z     
    Eu_2d, kh = average_over_discs_2D_v2(R_ii,kx, ky, kz)
    Eb_2d, kh = average_over_discs_2D_v2(B_k, kx, ky, kz)


    fig = plt.figure(figsize=(8,4), layout='constrained')

    ax1 = fig.add_subplot(121)
    ax1.semilogy(k, Eu_1d)
    ax1.set_xlabel(r'$|\mathbf{k}|$')
    ax1.set_ylabel(r'$E_U(|\mathbf{k}|,t)$')
    #ax1.set_ylim([1e-08, 1e-02])
    ax1.set_xlim([0, np.max(k)])
    
    ax2 = fig.add_subplot(122)
    for i, kz_i in enumerate(kz):
        if (i%5 == 0) and (i !=0):
            ax2.semilogy(kh, Eu_2d[:, i], label=r'$k_z = %d$' % kz_i)
    ax2.set_xlabel(r'$k_H$')
    ax2.set_ylabel(r'$E_U(k_H,k_z,t)$')
    #ax2.set_ylim([1e-16, 1e-02])
    ax2.set_xlim([0, np.max(k)])
    ax2.legend()

    fig.savefig('Kinetic Energy Spectra', dpi=100)
    plt.close(fig)

    # Create a 3D plot
    fig = plt.figure(figsize=(8,4), layout='constrained')

    ax1 = fig.add_subplot(121)
    ax1.semilogy(k, Eb_1d)
    ax1.set_xlabel(r'$|\mathbf{k}|$')
    ax1.set_ylabel(r'$E_B(|\mathbf{k}|,t)$')
    #ax1.set_ylim([1e-08, 1e-02])
    ax1.set_xlim([0, np.max(k)])
    
    ax2 = fig.add_subplot(122)
    for i, kz_i in enumerate(kz):
        if (i%5 == 0) and (i !=0):
            ax2.semilogy(kh, Eb_2d[:, i], label=r'$k_z = %d$' % kz_i)
    ax2.set_xlabel(r'$k_H$')
    ax2.set_ylabel(r'$E_B(k_H,k_z,t)$')
    #ax2.set_ylim([1e-16, 1e-02])
    ax2.set_xlim([0, np.max(kh)])
    ax2.legend()

    fig.savefig('Buoyancy Energy Spectra', dpi=100)
    plt.close(fig)

    return None


def time_series(data_dir):
    """Plot the time-series of the Kinetic energy and buoyancy variance."""

    f  = h5py.File(data_dir + 'scalar_data/scalar_data_s1.h5', mode='r')
    
    # Shape time,x,y,z
    Eu     = f['tasks/Eu(t)'][:,0,0,0]
    Eb     = f['tasks/Eb(t)'][:,0,0,0]
    wB_avg = f['tasks/<wB>'][:,0,0,0]
    B_avg  = f['tasks/<B>' ][:,0,0,0]
    t      = f['scales/sim_time'][()]
    
    f.close()

    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(8,4),constrained_layout=True)
    
    axs[0, 0].semilogy(t,Eu,'b-',label=r'$E_u$')
    axs[0, 0].set_title(r'$E_u$')
    axs[0, 1].semilogy(t,Eb,'r-',label=r'$E_b$')
    axs[0, 1].set_title(r'$E_b$')
    
    axs[1, 0].plot(t,wB_avg,'b:',label=r'$\langle wB \rangle$')
    axs[1, 0].set_title(r'$\langle wB \rangle$')
    axs[1, 1].plot(t, B_avg,'r:',label=r'$\langle B  \rangle$')
    axs[1, 1].set_title(r'$\langle B  \rangle$')
    
    fig.savefig('EnergyTimeSeries.png',dpi=100)
    plt.close(fig)

    return None


def pdfs(data_dir):
    """Save plot the joint pdf f_WB."""

    # Plot writes
    with h5py.File(data_dir + "/snapshots/snapshots_s1.h5", mode='r') as file:
        
        t = file['tasks/b'].dims[0][0][:]
        pad = int(len(t)//5)
        x = file['tasks/b'].dims[1][0][:]
        y = file['tasks/b'].dims[2][0][:]
        z = file['tasks/b'].dims[3][0][:]

        w_split = [] 
        b_split = []
        for i in range(pad, len(t), 1):
            w = file['tasks/w'][i, :, :, :]
            b = file['tasks/b'][i, :, :, :]
            
            # Interpolate onto a equispaced grid
            w_split.append( w.flatten() )
            b_split.append( b.flatten() )

    W = np.concatenate(w_split)
    B = np.concatenate(b_split)

    f_WB, w_edges, b_edges = np.histogram2d(W, B, bins=(128, 128), density=True)
    w = .5*(w_edges[1:] + w_edges[:-1])
    b = .5*(b_edges[1:] + b_edges[:-1])

    dw = w[1]-w[0]
    db = b[1]-b[0]

    f_W = np.nansum(f_WB, axis=1)*db
    f_B = np.nansum(f_WB, axis=0)*dw

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,6), layout='constrained')

    # First column
    ax[0].set_ylabel('$f_B$', fontsize=25)
    ax[0].set_xlabel('$b$', fontsize=25)
    #ax[0,0].plot(x, f_X, 'r-')
    ax[0].stairs(f_B, b_edges)
    ax[0].fill_between(x=b,y1=f_B,color= "r",alpha= 0.2)
    ax[0].set_ylim([0,1.1*max(f_B)])

    ax[1].set_title(r'$f_{WB}$', fontsize=25)
    cf = ax[1].pcolormesh(b,w,f_WB, cmap='Reds')#, norm='log')
    #fig.colorbar(cf, ax=ax1)
    ax[1].set_xlabel('$b$', fontsize=25)
    ax[1].set_ylabel('$w$', fontsize=25)

    ax[2].set_xlabel('$w$', fontsize=25)
    ax[2].set_ylabel('$f_W$', fontsize=25)
    #ax[2].plot(f_W, w, 'r-')
    ax[2].stairs(f_W, w_edges)
    ax[2].fill_between(x=w,y1=f_W,color= "r",alpha= 0.2)
    ax[2].set_ylim([0,1.1*max(f_W)])
    
    fig.savefig('PDF_Lundgren_f_WB.png', dpi=100)
    plt.show()

    return None


def plot_pdfs(data_dir, Nlevels=50, norm='log'):
    """Return a base plot of the PDFs ontop of which we can overlay the expectations."""

    # Plot writes
    with h5py.File(data_dir + "/snapshots/snapshots_s1.h5", mode='r') as file:
        
        t = file['tasks/b'].dims[0][0][:]
        pad = int(len(t)//5)
        x = file['tasks/b'].dims[1][0][:]
        y = file['tasks/b'].dims[2][0][:]
        z = file['tasks/b'].dims[3][0][:]

        w_split = [] 
        b_split = []
        for i in range(pad, len(t), 1):
            w = file['tasks/w'][i, :, :, :]
            b = file['tasks/b'][i, :, :, :]
            
            # Interpolate onto a equispaced grid
            w_split.append( w.flatten() )
            b_split.append( b.flatten() )

    W = np.concatenate(w_split)
    B = np.concatenate(b_split)

    f_WB, w_edges, b_edges = np.histogram2d(W, B, bins=(64, 64), density=True)
    w = .5*(w_edges[1:] + w_edges[:-1])
    b = .5*(b_edges[1:] + b_edges[:-1])

    dw = w[1]-w[0]
    db = b[1]-b[0]

    f_W = np.nansum(f_WB, axis=1)*db
    f_B = np.nansum(f_WB, axis=0)*dw
    from matplotlib.ticker import NullFormatter, MaxNLocator
    from numpy import linspace
    #plt.ion()


    # Coords
    x = w
    y = b

    # Data
    f_x  = f_W
    f_y  = f_B
    f_xy = f_WB

    # Set up your x and y labels
    xlabel = r'$w$'
    ylabel = r'$b$'

    fxlabel = r'$f_W(w)$'
    fylabel = r'$f_B(b)$'

    # Set up default x and y limits
    xlims = [min(x),max(x)]
    ylims = [min(y),max(y)]
    
    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02
    
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
    
    # Set up the size of the figure
    fig = plt.figure(1, figsize=(12,9))
    
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx       = plt.axes(rect_histx) # x histogram
    axHisty       = plt.axes(rect_histy) # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # Find the min/max of the data
    xmin = min(xlims)
    xmax = max(xlims)
    ymin = min(ylims)
    ymax = max(y)

    # Make the 'main' 2D plot
    axTemperature.contourf(x, y, f_xy.T, cmap='Reds', levels=Nlevels, norm=norm)
    axTemperature.set_xlabel(xlabel,fontsize=30)
    axTemperature.set_ylabel(ylabel,fontsize=30)

    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(30)
        label.set_family('serif')
    
    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(30)
        label.set_family('serif')
    
    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)

    # Make the 1D plots
    axHistx.plot(x,f_x,'r')
    axHistx.set_ylim([0.,1.01*max(f_x)])
    axHistx.fill_between(x=x,y1=f_x,color= "r",alpha= 0.2)
    
    axHisty.plot(f_y,y,'r')
    axHisty.set_xlim([0.,1.01*max(f_y)])
    axHisty.fill_between(x=f_y,y1=y,color= "r",alpha= 0.2)
    
    axHistx.set_ylabel(fxlabel,fontsize=30)
    axHisty.set_xlabel(fylabel,fontsize=30)

    #Set up the histogram limits
    axHistx.set_xlim( min(x), max(x) )
    axHisty.set_ylim( min(y), max(y) )
    
    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(30)
        label.set_family('serif')
    
    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(30)
        label.set_family('serif')
    
    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHisty.yaxis.set_major_locator(MaxNLocator(6))
    axTemperature.yaxis.set_major_locator(MaxNLocator(6))

    axHistx.yaxis.set_major_locator(MaxNLocator(4))
    axHistx.xaxis.set_major_locator(MaxNLocator(6))
    axTemperature.xaxis.set_major_locator(MaxNLocator(6))
    
    fig.savefig('PDF_Lundgren_f_WB.png', dpi=100)
    plt.close(fig)

    return None


if __name__ == "__main__":

    data_dir = "./"
    #video("./")
    #main("./")
    #time_series("./")
    energy_spectrum("./")
    #plot_pdfs("./")
