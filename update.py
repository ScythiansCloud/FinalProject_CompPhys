'''implementation of the algorithm'''
import force
import numpy as np


def update(USEFORCE, x,y,z,vx,vy,vz, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kbT, xi, delta_t, gaus_var):

    zeta_x = np.random.normal(loc=0, scale=gaus_var)
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
        ax, ay, az = 0,0,0

    # update velocities
    
    vx += ax*delta_t + np.sqrt(2*kbT*xi)/m*zeta_x* np.sqrt(delta_t)
    vy += ay*delta_t + np.sqrt(2*kbT*xi)/m*zeta_y* np.sqrt(delta_t)
    vz += az*delta_t + np.sqrt(2*kbT*xi)/m*zeta_z* np.sqrt(delta_t)

    # update positions  
    x += vx * delta_t
    y += vy * delta_t
    z += vz * delta_t  ### noch PBC adden (oder vielleicht sollten wir das nur in der darstellung machen dann umd die MSD zu berechnen)

    return x,y,z,vx,vy,vz

    