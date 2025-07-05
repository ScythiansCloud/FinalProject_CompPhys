import numpy as np

__all__ = ["compute_msd", "plot_msd"]


def compute_msd(positions, max_lag=None, box_length=None):
    """Vectorised Mean‑Squared Displacement (MSD)

    Parameters
    ----------
    positions : ndarray, shape (n_frames, n_particles, dim)
        Un‑wrapped particle coordinates.
    max_lag : int, optional
        Maximum lag *in saved snapshots* for which the MSD is evaluated.
        Defaults to ``n_frames // 2`` (recommended in the assignment).
    box_length : float or array_like, optional
        If given, periodic boundary conditions are assumed and the minimum‑
        image convention is applied before squaring the displacement.

    Returns
    -------
    lags : ndarray, shape (max_lag,)
        Integer time lags (units: snapshots).
    msd : ndarray, shape (max_lag,)
        ⟨|r(t'+t) − r(t')|²⟩ averaged over *all* particles and time origins.
    """

    # Ensure a floating, contiguous array — **Numba‑friendly** dtype spec
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    if positions.ndim != 3:
        raise ValueError("`positions` must have shape (n_frames, n_particles, dim)")

    n_frames, n_particles, dim = positions.shape

    if max_lag is None:
        max_lag = n_frames // 2
    if max_lag < 1 or max_lag >= n_frames:
        raise ValueError("`max_lag` must satisfy 1 ≤ max_lag < n_frames")

    # Box length handling for PBCs
    if box_length is not None:
        box_length = np.asarray(box_length, dtype=np.float64)
        if box_length.ndim == 0:
            box_length = np.full(dim, box_length, dtype=np.float64)
        elif box_length.shape != (dim,):
            raise ValueError("`box_length` must be scalar or have shape (dim,)")

    lags = np.arange(1, max_lag + 1, dtype=np.int64)
    msd = np.empty_like(lags, dtype=np.float64)

    for i, lag in enumerate(lags):
        disp = positions[lag:] - positions[:-lag]
        if box_length is not None:
            disp -= np.round(disp / box_length) * box_length  # minimum‑image
        sq_disp = (disp ** 2).sum(axis=-1)  # over Cartesian components
        msd[i] = sq_disp.mean()            # ⟨…⟩ over particles & time origins

    return lags, msd


def plot_msd(ax, lags, msd, dt=1.0, **kwargs):
    """Quick Matplotlib helper.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    lags, msd : ndarray
        Output of :func:`compute_msd`.
    dt : float, optional
        Time per *saved* snapshot.
    **kwargs :
        Passed straight to :pycode:`ax.plot()`.
    """
    import matplotlib.pyplot as plt  # local import avoids hard dependency

    t = lags * dt
    line, = ax.plot(t, msd, **kwargs)
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$⟨r^2(t)⟩$")
    ax.set_title("Mean‑Squared Displacement")
    return line


if __name__ == "__main__":
    # Self‑test: free diffusion in 1‑D (D = 1) → MSD = 2Dt
    rng = np.random.default_rng(0)
    n_particles, n_frames, dt = 1000, 2000, 1.0
    steps = rng.normal(scale=np.sqrt(2 * dt), size=(n_frames, n_particles, 1))
    traj = np.cumsum(steps, axis=0)
    lags, msd = compute_msd(traj)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_msd(ax, lags, msd, dt)
    ax.plot(lags * dt, 2 * lags * dt, ls="--", label="Theory 2Dt")
    ax.legend()
    plt.show()
