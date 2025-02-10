"""
Prepares data in a Paraview readable format by changing from numpy to VTS filetype

To install the dependencies activate your conda environment 

conda activate dedalus3

and then run: 

pip install pyevtk

"""

import numpy as np
import h5py
from pyevtk.hl import gridToVTK


def Make_VTK(time_index, filename, OUT_FILE):
    """ 
    time_index, int - take first or -1 last
    DIR_ABS_PATH, string - Where to find the dedalus simulation data
    OUT_FILE, string - Name of vts file created
    """

    file = h5py.File(filename, 'r')     

    t = file['tasks/b'].dims[0][0][:]
    x = file['tasks/b'].dims[1][0][:]
    y = file['tasks/b'].dims[2][0][:]
    z = file['tasks/b'].dims[3][0][:]

    nx = len(x)
    ny = len(y)
    nz = len(z)

    #print("nx=", nx, "ny=", ny, "nz=", nz)

    s_shape = (nx, ny, nz)
    u = np.zeros(s_shape)
    v = np.zeros(s_shape)
    w = np.zeros(s_shape)
    B = np.zeros(s_shape)

    #u[:, :, :] = file['tasks/u'][time_index, :, :, :]
    #v[:, :, :] = file['tasks/v'][time_index, :, :, :]
    w[:, :, :] = file['tasks/w'][time_index, :, :, :]
    B[:, :, :] = file['tasks/b'][time_index, :, :, :]

    gridToVTK(OUT_FILE, x, y, z, pointData={"u": u, "v": v, "w": w,  "B": B})

    return None


if __name__ == "__main__":

    data_dir = "./"
    time_index = -1
    filename = data_dir + "/snapshots/snapshots_s1.h5"
    OUT_FILE = "frame" + "_t%i" % time_index

    Make_VTK(time_index, filename, OUT_FILE)

    # for time_index in range(100):
    #     if time_index%10 == 0:
    #         print('Time_index = ',time_index)
    #     OUT_FILE = "./DNS_M5e-05_T8_Re20Pm75" + "_t%i"%time_index;
    #     Make_VTK(time_index, filename, OUT_FILE);

    
