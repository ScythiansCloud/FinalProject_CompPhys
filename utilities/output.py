from settings import settings_task2
from settings import settings_task3
import numpy as np

def WriteTrajectory3d(fileoutput, itime, x, y, z, settings):
    
    fileoutput.write("ITEM: TIMESTEP \n")
    fileoutput.write("%i \n" % itime)
    fileoutput.write("ITEM: NUMBER OF ATOMS \n")
    fileoutput.write("%i \n" % (settings.N))
    fileoutput.write("ITEM: BOX BOUNDS \n")
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.L))
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.L))
    fileoutput.write("%e %e xlo xhi \n" % (0, settings.L))
    fileoutput.write("ITEM: ATOMS id type x y z \n")
    
    for i in range(0, settings.N):
        fileoutput.write("%i %i %e %e %e \n" % (i, i, x[i] % settings.L, y[i] % settings.L, z[i] % (settings.L)))

def WriteunwrappedState(fileoutput, itime, x,y,z,vx,vy,vz):
    N = len(x)

    fileoutput.write(str(itime))
    for i in range(N):
        fileoutput.write( ' '+ str(x[i])+' '+ str(y[i])+' '+ str(z[i])+' '+ str(vx[i])+' '+ str(vy[i])+' '+ str(vz[i]))
    fileoutput.write('\n')
     