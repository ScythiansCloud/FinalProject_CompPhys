import numpy as np
from numba import njit, prange

# __all__ = ["compute_msd", "plot_msd"]


import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_msd(positions):
    n_t, N, _ = positions.shape
    max_lag = n_t // 2
    msd = np.zeros(max_lag)
    lags = np.arange(1, max_lag + 1)

    for t in prange(1, max_lag + 1): 
        msd_t = 0.0

        for tprime in range(n_t - t): # changing the endpoint such that we dont ouverindex the positions 
            '''this way we can average over more point leading to a better converged msd'''
            for n in range(N):
                dx = positions[tprime + t, n, 0] - positions[tprime, n, 0]
                dy = positions[tprime + t, n, 1] - positions[tprime, n, 1]
                dz = positions[tprime + t, n, 2] - positions[tprime, n, 2]
                msd_t += dx*dx + dy*dy + dz*dz

        msd[t - 1] = msd_t / ((n_t - t) * N) #this also slightly changes the normalization (tmax is dependent on tlag now)

    return lags, msd

# # --------------------------------------------------------------------------
# #  A thin Python wrapper for validation & dtype conversion
# # --------------------------------------------------------------------------
# def compute_msd(positions, max_lag=None, box_length=None):
#     """
#     Mean-squared displacement without FFT, but JIT-accelerated with Numba.

#     Parameters
#     ----------
#     positions : float array, shape (n_frames, n_particles, dim)
#         **Unwrapped** coordinates.
#     max_lag   : int, optional
#         Longest time lag to evaluate.  Default = n_frames // 2.
#     box_length : None or float or sequence of floats
#         Box lengths for minimum-image convention.  If None → no PBCs.
#     """
#     # --- basic checks done outside Numba ----------------------------------
#     pos = np.ascontiguousarray(positions, dtype=np.float64)
#     if pos.ndim != 3:
#         raise ValueError("`positions` must have shape (n_frames, n_particles, dim)")

#     n_frames, _, dim = pos.shape
#     if max_lag is None:
#         max_lag = n_frames // 2
#     if not (1 <= max_lag < n_frames):
#         raise ValueError("`max_lag` must satisfy 1 ≤ max_lag < n_frames")

#     # broadcast / validate box length
#     if box_length is None:
#         bl = np.zeros(dim, dtype=np.float64)   # sentinel → no PBC correction
#     else:
#         bl = np.asarray(box_length, dtype=np.float64)
#         if bl.ndim == 0:
#             bl = np.full(dim, bl, dtype=np.float64)
#         elif bl.shape != (dim,):
#             raise ValueError("`box_length` must be scalar or have shape (dim,)")

#     # call the fast kernel
#     return _compute_msd_nb(pos, max_lag, bl)


# --------------------------------------------------------------------------
#  Fast kernel – nopython, parallel
# --------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _compute_msd_nb(pos, max_lag, box_len):
    n_frames, n_particles, dim = pos.shape
    msd   = np.empty(max_lag, dtype=np.float64)
    lags  = np.arange(1, max_lag + 1, dtype=np.int64)

    use_pbc = False
    for d in range(dim):
        if box_len[d] > 0.0:          # 0 means “no PBC” (see wrapper)
            use_pbc = True
            break

    for i in prange(max_lag):        # prange over integer range → OK
        lag   = i + 1
        accum = 0.0
        norm  = (n_frames - lag) * n_particles   # number of displacement vectors

        for t in range(n_frames - lag):
            for p in range(n_particles):
                # displacement vector
                dx = pos[t + lag, p, 0] - pos[t, p, 0]
                dy = pos[t + lag, p, 1] - pos[t, p, 1]
                dz = pos[t + lag, p, 2] - pos[t, p, 2]

                if use_pbc:
                    if box_len[0] > 0.0:
                        dx -= np.round(dx / box_len[0]) * box_len[0]
                    if dim > 1 and box_len[1] > 0.0:
                        dy -= np.round(dy / box_len[1]) * box_len[1]
                    if dim > 2 and box_len[2] > 0.0:
                        dz -= np.round(dz / box_len[2]) * box_len[2]

                accum += dx*dx + dy*dy + dz*dz

        msd[i] = accum / norm

    return lags, msd



def plot_msd(ax, lags, msd, dt=1.0, **kwargs):
    t = lags * dt
    line, = ax.plot(t, msd, **kwargs)
    ax.set_xlabel(r"Time $t\,[\tau_{\mathrm{LD}}]$")
    ax.set_ylabel(r"$\langle r^{2}(t) \rangle\,[\sigma^{2}]$")
    ax.set_title("Mean‑Squared Displacement")
    return line


# if __name__ == "__main__":
#     # Self‑test: free diffusion in 1‑D (D = 1) → MSD = 2Dt
#     rng = np.random.default_rng(0)
#     n_particles, n_frames, dt = 1000, 2000, 1.0
#     steps = rng.normal(scale=np.sqrt(2 * dt), size=(n_frames, n_particles, 1))
#     traj = np.cumsum(steps, axis=0)
#     lags, msd = compute_msd(traj)

#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     plot_msd(ax, lags, msd, dt)
#     ax.plot(lags * dt, 2 * lags * dt, ls="--", label="Theory 2Dt")
#     ax.legend()
#     plt.show()
