'''idk what ur opninion is here we could go for some random initialization
 (maybe place them one by one such that they do not overlap) or on a grid (i think its not specified in the exercies)
 
 Grid doesnt work that well because N doest have a 3rd root :/
 '''


# imports
import numpy as np
import settings_task2 as settings
#import settings_task3 as settings
from numba import njit

 
def InitializeAtoms(Csi):
    settings.init(Csi)
    

    n = 0
    x = np.zeros(shape=(settings.N))
    y = np.zeros(shape=(settings.N))
    z = np.zeros(shape=(settings.N))
    vx = np.zeros(shape=(settings.N))
    vy = np.zeros(shape=(settings.N))
    vz = np.zeros(shape=(settings.N))

    i =   0
    while n < settings.N:
        x0 = np.random.rand()*settings.L
        y0 =  np.random.rand()*settings.L
        z0 =  np.random.rand()*settings.L

        b = False
        for i in range(n):
            rijx = pbc(x[i], x0, 0, settings.L)
            rijy = pbc(y[i], y0, 0, settings.L)
            rijz = pbc(z[i], z0, 0, settings.L)
            
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            r = np.sqrt(r2)

            if r < settings.sig:
                b = True #reject placement
            
        if not b:
            x[n] = x0
            y[n] = y0
            z[n] = z0

            

            vx0 = 0.5 - np.random.rand()
            vy0 = 0.5 - np.random.rand()
            vz0 = 0.5 - np.random.rand()
            
            vx[n] = vx0
            vy[n] = vy0
            vz[n] = vz0

            n+= 1
        else:
            i +=1
            pass # tries same n again

        if i> settings.N**3:
            print('could not find enough positions, density to high')
            break

    
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    
    vx -= svx / settings.N  # type: ignore
    vy -= svy / settings.N 
    vz -= svz / settings.N 
    # svx = np.sum(vx)
    
    # rescale the velocity to the desired temperature
    kbT_random = Thermalenergy(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, settings.kBT, kbT_random)
    
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    
    vx -= svx / settings.N 
    vy -= svy / settings.N 
    vz -= svz / settings.N 
    
    return x, y, z, vx, vy, vz

@njit  
def pbc(xi, xj, xlo, xhi):
    
    l = xhi-xlo
    
    xi = xi % l
    xj = xj % l
    
    rij = xj - xi  
    if abs(rij) > 0.5*l:
        rij = rij - np.sign(rij) * l 
        
    return rij


def Thermalenergy(vx, vy, vz):
    vsq = np.sum(vx*vx + vy*vy + vz*vz)

    return vsq/settings.N*settings.m/3 # == kbT
    
def rescalevelocity(vx, vy, vz, kbT1, kbT2):
    fac = np.sqrt(kbT1 / kbT2)  # T1 is desired temperature
    vx = vx * fac
    vy = vy * fac
    vz = vz * fac
    return vx, vy, vz      

if __name__ == '__main__':
    settings.init(10)
    #print(InitializeAtoms()[0])

    import matplotlib.pyplot as plt
    x, y, z, _, _, _ = InitializeAtoms(10)
    # plt.figure(figsize=[10,20])
    plt.scatter(y,z)
    #plt.scatter(settings.deltaxyz/2, settings.deltaxyz/2)
    # plt.scatter(settings.deltaxyz, settings.deltaxyz-settings.bond_len/2)
    plt.xlim([0, settings.L])
    plt.ylim([0, settings.L])
    plt.show()                  
    

    # test
    
    
