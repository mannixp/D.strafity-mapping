"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


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
        quad = ax.pcolormesh(x, z, b[0, :, :].T, cmap='RdBu')
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


def spectra(data_dir):
    """Plot the time-averaged spectra of the Kinetic energy and buoyancy variance."""


    f  = h5py.File(data_dir + 'scalar_data/scalar_data_s1.h5', mode='r')
    
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4),constrained_layout=True)
   
    ax1.semilogy(f['tasks/Eu(kx)'][-1,:,0,0] ,'r.')
    ax1.set_ylabel(r'Kinetic Energy')
    ax1.set_xlabel(r'Fourier mode kx')

    ax2.semilogy(f['tasks/Eu(kz)'][-1,0,0,:],'b.')
    ax2.set_ylabel(r'Kinetic Energy')
    ax2.set_xlabel(r'Fourier mode kz')
    fig.savefig('Kinetic Energy Spectra', dpi=100)
    plt.close(fig)
    
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(8,4))
    ax1.semilogy(f['tasks/Eb(kx)'][-1,:,0,0] ,'r.')
    ax1.set_ylabel(r'Buoyancy Energy')
    ax1.set_xlabel(r'Fourier mode kx')

    ax2.semilogy(f['tasks/Eb(kz)'][-1,0,0,:],'b.')
    ax2.set_ylabel(r'Buoyancy Energy')
    ax2.set_xlabel(r'Fourier mode kz')
    fig.savefig('Buoyancy Energy Spectra', dpi=100)
    plt.close(fig)


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

    return None;


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
    video("./")
    main("./")
    spectra("./")
    plot_pdfs("./")
