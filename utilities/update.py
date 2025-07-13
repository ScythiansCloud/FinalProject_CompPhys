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


def compute_S_of_k_from_gr(g_of_r: np.ndarray,dr: float, rho: float, k: np.ndarray,) -> np.ndarray:
    """
    Given:
      - g_of_r:  1D array of length nbins, the time‐averaged RDF g(r) for r in [0, L/2)
      - dr:      bin width used to build g(r)
      - rho:     number density N/V
    Returns:
      - S(k):    structure factor as a function of k
    """
    #determining the center radii of the bins
    r_centers = (np.arange(len(g_of_r)) + 0.5) * dr
    # sinc function with exception for k == 0
    sinc      = 1.0 if k == 0 else np.sin(k*r_centers)/(k*r_centers)
    #integration
    S_of_k    = 1 + 4*np.pi*rho*np.sum((g_of_r-1)*r_centers**2*sinc*dr)
    return S_of_k


def calcg(Ngr, hist, dr, rho, N):
    r= r = np.arange(len(hist)) * dr    # hist goes up to l/2
    nid = 4*np.pi *rho /3 * ((r+ dr)**3-r**3) # type: ignore
    n = hist/ N /Ngr
    return n/nid


@njit(parallel=True)
def update_hist(hist, x,y,z, dr, N, L):

    for i in prange(N-1):
        for j in range(i+1,N):
            rijx = force.pbc(x[i], x[j], 0, L) # calculate pbc distance
            rijy = force.pbc(y[i], y[j], 0, L)
            rijz = force.pbc(z[i], z[j], 0, L)
            r = np.sqrt(rijx**2 + rijy**2 + rijz**2)
            if r > 0.0 and r < L/2:
                bin = int(r / dr) # find the bin
                hist[bin] += 2 # we are counting pairs
    return hist