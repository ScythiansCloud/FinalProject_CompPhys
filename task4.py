#!/usr/bin/env python3
"""Task 4 – Analyse trajectories (g(r), S(k), MSD)

* Re‑uses the **latest** `output/run_X` directory created by Tasks 2/3.
* Reads the wrapped (and, if present, unwrapped) trajectories it finds
  there and writes three analysis figures back to the same folder:

  | file                  | contents                              |
  |-----------------------|---------------------------------------|
  | `rdf.png`             | radial‑distribution function *g(r)*   |
  | `structure_factor.png`| static structure factor *S(k)*        |
  | `msd.png`             | mean‑squared displacement (optional)  |

Just open *task4.py* in VS Code and press ▶ (or run `python task4.py`).
"""
from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utilities import update           # Numba‑accelerated RDF kernel
from utilities import utils            # logging helpers
from utilities.msd import compute_msd, plot_msd
import settings.settings_task4 as cfg  # dr, kmax, nk, everyN

# -----------------------------------------------------------------------------
# Optional import of Task‑3 simulation constants (kBT, xi, …)
# -----------------------------------------------------------------------------
try:
    sim_cfg = importlib.import_module("settings.settings_task3")
except ModuleNotFoundError:  # allow running even if the file is absent
    class _Dummy:  # minimal stand‑in with no attributes
        pass
    sim_cfg = _Dummy()  # type: ignore

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def latest_run_directory(root: Path) -> Path:
    """Return the run_* directory with the highest index."""
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise RuntimeError("No run directory found – generate trajectories first.")
    runs.sort(key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)))
    return runs[-1]


def find_traj_files(run_dir: Path) -> tuple[Path, Path | None]:
    """Pick the newest wrapped and (optionally) unwrapped trajectory files."""
    files = sorted(run_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    wrapped = next((p for p in files if "unwrapped" not in p.name.lower() and "nsteps" in p.name.lower()), None)
    unwrapped = next((p for p in files if "unwrapped" in p.name.lower()), None)
    if wrapped is None:
        raise RuntimeError(f"No wrapped trajectory containing 'nsteps' found in {run_dir}")
    return wrapped, unwrapped

# -----------------------------------------------------------------------------
# Lightweight file readers
# -----------------------------------------------------------------------------

def read_lammpstraj(path: Path) -> tuple[np.ndarray, float]:
    """Read ASCII trajectory written by WriteTrajectory3d (wrapped coords)."""
    frames: list[np.ndarray] = []
    L: float | None = None
    with path.open() as fh:
        while True:
            if not fh.readline():        # ITEM: TIMESTEP  (or EOF)
                break
            fh.readline()               # timestep value (unused)
            fh.readline()               # ITEM: NUMBER OF ATOMS
            natoms = int(fh.readline())
            fh.readline()               # ITEM: BOX BOUNDS
            xlo, xhi, *_ = map(float, fh.readline().split()[:2])
            ylo, yhi, *_ = map(float, fh.readline().split()[:2])
            zlo, zhi, *_ = map(float, fh.readline().split()[:2])
            if L is None:
                L = xhi - xlo
            fh.readline()               # ITEM: ATOMS
            coords = np.fromfile(fh, count=natoms * 5, sep=" ").reshape(natoms, 5)[:, 2:5]
            frames.append(coords)
    if not frames:
        raise ValueError(f"No frames in {path}")
    assert L is not None
    return np.asarray(frames, float), float(L)


def read_unwrapped(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    natoms = (data.shape[1] - 1) // 6
    return data[:, 1:1 + 3 * natoms].reshape(data.shape[0], natoms, 3)

# -----------------------------------------------------------------------------
# Analysis kernels
# -----------------------------------------------------------------------------

def compute_rdf(positions: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    n_frames, N, _ = positions.shape
    nbins = int((L / 2) / cfg.dr)
    hist = np.zeros(nbins, float)
    for xyz in positions:
        hist = update.update_hist(hist, xyz[:, 0], xyz[:, 1], xyz[:, 2], cfg.dr, N, L)
    rho = N / L**3
    g_r = update.calcg(n_frames, hist, cfg.dr, rho, N)
    r = (np.arange(nbins) + 0.5) * cfg.dr
    return r, g_r


def first_peak_and_coordination(r: np.ndarray, g: np.ndarray, rho: float) -> tuple[float, float]:
    peak_idx = int(np.argmax(g))
    try:
        cut_idx = peak_idx + 1 + np.where(g[peak_idx + 1:] < 1.0)[0][0]
    except IndexError:
        cut_idx = len(g) - 1
    dr = r[1] - r[0]
    coord = 4 * np.pi * rho * np.sum(g[:cut_idx + 1] * r[:cut_idx + 1] ** 2) * dr
    return g[peak_idx], coord


def compute_structure_factor(r: np.ndarray, g: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    kmin = 2 * np.pi / r.max() / 2
    k = np.linspace(kmin, cfg.kmax * kmin, cfg.nk)
    dr = r[1] - r[0]
    gr1 = g - 1
    S = np.empty_like(k)
    for i, kk in enumerate(k):
        sinc = np.sin(kk * r) / (kk * r)
        S[i] = 1 + 4 * np.pi * rho * np.sum(gr1 * r**2 * sinc) * dr
    return k, S


def analyse_msd(unwrapped: np.ndarray, dt_snap: float) -> tuple[np.ndarray, np.ndarray, float]:
    lags, msd = compute_msd(unwrapped, max_lag=len(unwrapped) // 2)
    t = lags * dt_snap
    slope, _ = np.polyfit(t[int(0.7 * len(t)):], msd[int(0.7 * len(t)):], 1)
    D_est = slope / 6
    return t, msd, D_est

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    run_dir = latest_run_directory(ROOT / "output")

    utils.setup_logging(run_dir)
    logging.info("=== Task 4 analysis started in %s ===", run_dir.name)

    wrapped_file, unwrapped_file = find_traj_files(run_dir)
    logging.info("Wrapped trajectory: %s", wrapped_file.name)
    if unwrapped_file:
        logging.info("Unwrapped trajectory: %s", unwrapped_file.name)

    # Load trajectories
    positions_wrapped, L = read_lammpstraj(wrapped_file)
    rho = positions_wrapped.shape[1] / L**3

    # g(r)
    r, g_r = compute_rdf(positions_wrapped, L)
    peak, coord = first_peak_and_coordination(r, g_r, rho)
    logging.info("g(r): peak %.3f  coordination %.2f", peak, coord)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(r, g_r)
    ax.set_xlabel(r"$r/\sigma$")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial distribution function")
    ax.axhline(1, color="0.6", ls=":")
    fig.tight_layout()
    (run_dir / "rdf.png").write_bytes(fig.canvas.buffer_rgba())
    fig.savefig(run_dir / "rdf.png", dpi=300)
    plt.close(fig)

    # S(k)
    k_vals, S_k = compute_structure_factor(r, g_r, rho)

    # Optional compressibility if kBT known
    kBT = getattr(sim_cfg, "kBT", None) or getattr(sim_cfg, "kB", None) and getattr(sim_cfg, "T", None) and getattr(sim_cfg, "kB") * getattr(sim_cfg, "T")
    if kBT:
        kBT = float(kBT)
        logging.info("Isothermal compressibility κ_T ≈ %.3f", S_k[0] / (rho * kBT))
    else:
        logging.info("kBT not available – skipping κ_T estimate")

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(k_vals, S_k)
    ax.set_xlabel("k")
    ax.set_ylabel("S(k)")
    ax.set_title("Structure factor")
    fig.tight_layout()
    fig.savefig(run_dir / "structure_factor.png", dpi=300)
    plt.close(fig)

    # MSD
    if unwrapped_file is not None:
        unwrapped = read_unwrapped(unwrapped_file)
        dt_snap = cfg.everyN * getattr(sim_cfg, "delta_t", 1.0)
        t, msd, D_est = analyse_msd(unwrapped, dt_snap)
        logging.info("D_fit ≈ %.3g", D_est)

        xi = getattr(sim_cfg, "xi", None)
        kBT_for_D = getattr(sim_cfg, "kBT", None)
        D_th = kBT_for_D / xi if (xi is not None and kBT_for_D is not None and xi != 0) else None
        if D_th:
            logging.info("kBT/ξ (= %.3g) provided – comparing to theory", D_th)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        plot_msd(ax, t, msd, label="simulation")

        if D_th is not None:
            ax.plot(t, 6 * D_th * t, ls="--", label="6Dt (free)")
        ax.set_title("Mean‑squared displacement")
        ax.set_xlabel("t (simulation units)")
        ax.set_ylabel(r"$\langle r^{2}(t) \rangle$")
        if D_th is not None:
            ax.legend()
        msd_png = run_dir / "msd.png"
        fig.tight_layout()
        fig.savefig(msd_png, dpi=300)
        plt.close(fig)
        logging.info("Saved %s", msd_png.name)
    else:
        logging.info("No unwrapped trajectory – MSD skipped.")

    logging.info("=== Task 4 analysis finished ===")


if __name__ == "__main__":
    main()
