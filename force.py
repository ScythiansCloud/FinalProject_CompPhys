'''here we will calculate the forces'''

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def acc(x, y, z, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kbT):

# Wir geben die force zurück aber die Funktion heißt acc, finde ich etwas verwirrend
    
    fx = np.zeros(shape=len(x))
    fy = np.zeros(shape=len(x))
    fz = np.zeros(shape=len(x))



    for i in range(N-1):
        for j in range(i+1, N):

            rijx = pbc(x[i], x[j], 0, L)
            rijy = pbc(y[i], y[j], 0, L)
            rijz = pbc(z[i], z[j], 0, L)
            
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            r = np.sqrt(r2)

            if r > (sig + delta):       # Jonas: Denke wir könnten hier auf noch einen cutoff hinzufügen, 
                                        # damit wir nicht zu viele Kräfte berechnen müssen
                '''calculate vdw + elec interaction'''

                VDW = A*sig/(24*m*(r-sig)**2)   # Jonas: habe ich auch so aber ohne das m. Warum sollte das da sein?
                EL = -Zprimesqrd*lambda_B*np.exp(-kappa_D*r)*(1/r+kappa_D)/(m*r**2) # Jonas: habe ich auch so nur wieder ohne masse m
                SUM = VDW + EL

                fx[i] -= SUM* rijx
                fy[i] -= SUM* rijy
                fz[i] -= SUM* rijz
                fx[j] += SUM* rijx
                fy[j] += SUM* rijy
                fz[j] += SUM* rijz
            
            else:
                '''calculate lj interacion'''
                # Jonas: jo passt mMn
                LJ = 8*kbT/m*((0.27993600/r**8-0.02612138803199999/r**14))

                fx[i] -= LJ* rijx
                fy[i] -= LJ* rijy
                fz[i] -= LJ* rijz
                fx[j] += LJ* rijx
                fy[j] += LJ* rijy
                fz[j] += LJ* rijz
    return fx, fy, fz               


### doppelcheckt mla jmd die ableitung der potentiale ob das so sinn macht und algebraisch auch stimmt (und vorzeichen bin ihc mir nicht 100% sicher aber ist mal ein anfang :)



@njit  
def pbc(xi, xj, xlo, xhi):
    
    l = xhi-xlo
    
    xi = xi % l
    xj = xj % l
    
    rij = xj - xi  
    if abs(rij) > 0.5*l:
        rij = rij - np.sign(rij) * l 
        
    return rij


@njit(parallel=True)
def berendsen_thermostat(vx, vy, vz, T, T0, delta_t, tau_berendsen):
    """
    tau:    coupling strength
    T:      current Temperature
    T0:     Temp. to jump to / approach
    """
    lambdaa = np.sqrt(1 + delta_t / tau_berendsen * (T0/T - 1))
    for i in prange(len(vx)):
        vx[i] *= lambdaa
        vy[i] *= lambdaa
        vz[i] *= lambdaa



if __name__ == '__main__':
    import numpy as np
    for i in range(50):
        print(np.random.rand())