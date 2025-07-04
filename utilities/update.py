'''implementation of the algorithm'''
import utilities.force as force
import numpy as np
from numba import njit, prange

# @njit(parallel=True)
# @njit
def update(TASK2, x,y,z,vx,vy,vz, L, N, sig, delta, A, m, Zprimesqrd, 
           lambda_B, kappa_D, kBT, xi, delta_t, gaus_var, random_seed):

    # need to place the random seed outside the call of update
    # added here 
    zeta_x = np.random.normal(loc=0, scale=gaus_var, size=N)  # don't know how to use a random_seed with this function here...
    zeta_y = np.random.normal(loc=0, scale=gaus_var, size=N)    
    zeta_z = np.random.normal(loc=0, scale=gaus_var, size=N)


    if TASK2: # depending on task 2 or 3 we do not need the interaction forces
        # no forces, only friction
        ax = -xi*vx/m
        ay = -xi*vy/m
        az = -xi*vz/m
        
    else:
        fx, fy, fz, x, y, z = force.acc(x, y, z, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kBT)
        # add friction
        ax = fx/m-xi*vx/m        
        ay = fy/m-xi*vy/m
        az = fz/m-xi*vz/m

    # update velocities
    coeff = np.sqrt(2 * kBT * xi) / m * np.sqrt(delta_t)
    vx += ax*delta_t + coeff * zeta_x
    vy += ay*delta_t + coeff * zeta_y
    vz += az*delta_t + coeff * zeta_z

    # update positions
    x += vx * delta_t
    y += vy * delta_t
    z += vz * delta_t

    return x,y,z,vx,vy,vz

    