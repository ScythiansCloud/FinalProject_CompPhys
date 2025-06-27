'''implementation of the algorithm'''
import force
import numpy as np


def update(USEFORCE, x,y,z,vx,vy,vz, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kbT, xi, delta_t, gaus_var):
    zeta = np.random.normal(loc=0, scale=gaus_var)
    if USEFORCE:
        fx, fy, fz = force.acc(x, y, z, L, N, sig, delta, A, m, Zprimesqrd, lambda_B, kappa_D, kbT)

        # add friction
        ax = fx-xi*vx/m
        ay = fy-xi*vy/m
        az = fz-xi*vz/m
    else:
        ax, ay, az = 0,0,0

    # update velocities
    randomvel = np.sqrt(2*kbT*xi)/m*zeta* np.sqrt(delta_t) # verwirrt mich ein wenig weil die ja dan immer nur in die (1,1,1) richtung zeigt idk?
    
    vx += ax*delta_t + randomvel
    vy += ay*delta_t + randomvel
    vz += az*delta_t + randomvel

    # update positions  
    x += vx * delta_t
    y += vy * delta_t
    z += vz * delta_t

    return x,y,z,vx,vy,vz

    