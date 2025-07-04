'''here we define functions that can calculate the MSD of a give trajectory'''


from numba import njit, prange
import numpy as np

def dist(x1,y1,z1,x0,y0,z0):
    return 1

def MSD(x,y,z):

    N = len(x[0])

    steps = len(x)

    lagtime = steps//2 #maximum lagtime
    tmax = steps //2 # different starting point 


    rsqrd = np.zeros(steps)

    for i in prange(N):
        for t in prange(lagtime):
            for t_prime in range(tmax):
                rsqrd[t] += dist(x[t_prime+t,i],y[t_prime+t,i],z[t_prime+t,i], x[t_prime,i], y[t_prime,i], z[t_prime,i]) **2

            



            

