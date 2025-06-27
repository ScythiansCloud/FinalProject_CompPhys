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

            n+= 1

            vx0 = 0.5 - np.random.rand()
            vy0 = 0.5 - np.random.rand()
            vz0 = 0.5 - np.random.rand()
            
            vx[n] = vx0
            vy[n] = vy0
            vz[n] = vz0
        else:
            i +=1
            pass # tries same n again


        if i> settings.N**3:
            n = settings.N #implement a saveguard so that the while loop does not go forever
            print('could not find enough positions, density to high')

    
    # cancel the linear momentum
    svx = np.sum(vx)
    svy = np.sum(vy)
    svz = np.sum(vz)
    
    vx -= svx / settings.N  # type: ignore
    vy -= svy / settings.N 
    vz -= svz / settings.N 
    # svx = np.sum(vx)
    
    # rescale the velocity to the desired temperature
    Trandom = temperature(vx, vy, vz)
    vx, vy, vz = rescalevelocity(vx, vy, vz, settings.Tdesired, Trandom)
    
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


def temperature(vx, vy, vz):
    # receives units of [v] = nm/fs --> [v^2] = nm^2 /fs^2
    vsq = np.sum(vx*vx + vy*vy + vz*vz)

    # convert to kcal/mol:
    #   g/mol·(nm/fs)² → J/mol  by factor 1e9
    #   J/mol      → kcal/mol by dividing 4184
    conv = 1e9 / 4184.0

    K_kcal_per_mol = 0.5 * settings.mass * vsq * conv

    # equipartition: K = 3/2 N kB T  ⇒  T = 2K/(3N kB)
    return 2.0 * K_kcal_per_mol / (3.0 * settings.N * settings.kb)
    
def rescalevelocity(vx, vy, vz, T1, T2):
    fac = math.sqrt(T1 / T2)  # T1 is desired temperature
    vx = vx * fac
    vy = vy * fac
    vz = vz * fac
    return vx, vy, vz      

@njit(parallel=True)
def berendsen_thermostat(vx, vy, vz, T, T0, tau):
    """
    tau:    coupling strength
    T:      current Temperature
    T0:     Temp. to jump to / approach
    """
    lambdaa = np.sqrt(1 + settings.deltat / tau * (T0/T - 1))
    for i in prange(len(vx)):
        vx[i] *= lambdaa
        vy[i] *= lambdaa
        vz[i] *= lambdaa

    # return vx, vy, vz
    # no need to return anything since we just update the velocities


@njit(parallel=True)
def andersen_thermostat(vx, vy, vz, T0, nu):
    # vx, vy, vz : arrays of length N (particle velocities)
    # T0          : target temperature
    # nu           : collision frequency (collisions per unit time)

    p_collision = nu * settings.deltat   # probability each particle collides in this step
    # Precompute the Maxwell–Boltzmann velocity scale
    std = np.sqrt(settings.kb * T0 / settings.mass)

    for i in prange(settings.N):
        rdm = np.random.rand()
        if rdm < p_collision:
            # single components are normal distributed
            vx[i] = std * np.random.randn()
            vy[i] = std * np.random.randn()
            vz[i] = std * np.random.randn()

    # no need to return anything since we just update the velocities

if __name__ == '__main__':
    settings.init()
    #print(InitializeAtoms()[0])
    print(settings.deltaxyz)
    print(settings.bond_len)
    import matplotlib.pyplot as plt
    x, y, z, _, _, _ = InitializeAtoms()
    # plt.figure(figsize=[10,20])
    plt.scatter(y,z)
    #plt.scatter(settings.deltaxyz/2, settings.deltaxyz/2)
    # plt.scatter(settings.deltaxyz, settings.deltaxyz-settings.bond_len/2)
    plt.xlim([0, settings.l])
    plt.ylim([0, settings.l])
    plt.show()                  
    

    # test
    
    
