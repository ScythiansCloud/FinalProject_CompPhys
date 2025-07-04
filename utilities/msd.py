'''here we define functions that can calculate the MSD of a give trajectory'''


from numba import njit, prange
import numpy as np

def MSD(x,y,z):

    N = len(x[0])

    steps = len(x)

    rsqrd = np.zeros(steps)

    for t in prange(steps):
        for i in range(N):
            pass