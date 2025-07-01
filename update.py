'''implementation of the algorithm'''
import force
import numpy as np


def update(USEFORCE, x,y,z,vx,vy,vz, L, N, sig, delta, A, m, Zprimesqrd, 
           lambda_B, kappa_D, kbT, xi, delta_t, gaus_var, random_seed):

    np.random.seed(random_seed)
    zeta_x = np.random.normal(loc=0, scale=gaus_var)  # don't know how to use a random_seed with this function here...
    zeta_y = np.random.normal(loc=0, scale=gaus_var)
    zeta_z = np.random.normal(loc=0, scale=gaus_var)


    if USEFORCE: # depending on task 2 or 3 we need or do not need the acceleration
        fx, fy, fz = force.acc(x, y, z, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kbT)


        # add friction
        ax = fx-xi*vx/m         # weil acceleration = force/mass und m=1 ist eigentlich egal aber würde 
                                # trotzdem auch ax = fx/m-xi*vx/m etc. schreiben für die Zukunft
        ay = fy-xi*vy/m
        az = fz-xi*vz/m
    else:
        ax, ay, az = 0,0,0      # for task 2

    # update velocities
    vx += ax*delta_t + np.sqrt(2*kbT*xi)/m*zeta_x* np.sqrt(delta_t)     # passt denke ich
    vy += ay*delta_t + np.sqrt(2*kbT*xi)/m*zeta_y* np.sqrt(delta_t)
    vz += az*delta_t + np.sqrt(2*kbT*xi)/m*zeta_z* np.sqrt(delta_t)

    # update positions  
    x = (x + vx * delta_t) % L     # passt auch denke ich
    y = (y + vy * delta_t) % L
    z = (z + vz * delta_t) % L ### noch PBC adden (oder vielleicht sollten wir das nur in der darstellung machen dann umd die MSD zu berechnen)


    return x,y,z,vx,vy,vz

    