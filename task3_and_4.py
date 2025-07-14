# tasks3_and_4.py (debug fix 2 – 2025-07-12)
# ---------------------------------------------------------------------------
# Unified Task 3 + Task 4 b runner – fixed context-manager bug
# ---------------------------------------------------------------------------

"""Run Simulation3 for several salt concentrations → save g(r), S(k), κ_T.

This revision fixes *TypeError: 'tuple' object does not support the context
manager protocol* that appeared on Windows ≥ Py3.11 – we can’t put several
context managers inside parentheses.  Switched to **contextlib.nullcontext** so
we can still use a single `with` statement whether *write* is True or False.
Other small clean-ups:
• added missing imports (*initialize*, *update*, *contextlib.nullcontext*).
• clarified random-seed logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Imports and helper functions
# ---------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Sequence
from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")  # head-less backend – no GUI blocking
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Project-specific imports ----------------------------------------------------
from utilities import output, initialize, update  # low-level MD helpers
from utilities.simulation import Simulation3
from utilities.update import compute_S_of_k_from_gr  # scalar routine
import settings.settings_task3 as settings3
import settings.settings_task3_10 as settings3_Cs10
from utilities.utils import create_output_directory, setup_logging

# ---------------------------------------------------------------------------
# 1.a  Monky-patch Simulation3 (safer & faster) ------------------------------
# ---------------------------------------------------------------------------

def _Simulation3_patched(outdir, write, Traj_name, everyN, random_seed, settings):
    # --- random seed -------------------------------------------------------
    seed = settings.random_seed if random_seed is None else random_seed
    np.random.seed(seed)

    # --- initial configuration -------------------------------------------
    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, seed, settings)

    # one initial update so velocities are thermalised
    x, y, z, vx, vy, vz = update.update(
        False,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        settings.L,
        settings.N,
        settings.sig,
        settings.delta,
        settings.A,
        settings.m,
        settings.Zprimesqrd,
        settings.lambda_B,
        settings.kappa_D,
        settings.kBT,
        settings.xi,
        settings.delta_t,
        settings.gaus_var,
        seed,
    )

    # helper for filenames --------------------------------------------------
    def _fname(prefix: str):
        return outdir / f"{Traj_name}_{prefix}_everyN{everyN}_nsteps{settings.nsteps}"

    # choose real file handles or harmless nullcontext() stubs --------------
    cm_eq  = open(_fname("eq"), "w")             if write else nullcontext()
    cm_equ = open(_fname("eq_unwrapped"), "w")    if write else nullcontext()
    cm_pr  = open(_fname("prod"), "w")           if write else nullcontext()
    cm_pru = open(_fname("prod_unwrapped"), "w")  if write else nullcontext()

    # open all four in one WITH statement (comma-separated list) ------------
    with cm_eq  as file_eq, \
         cm_equ as file_eq_unw, \
         cm_pr  as file_prod, \
         cm_pru as file_prod_unw:

        if write:
            output.WriteTrajectory3d(file_eq, 0, x, y, z, settings)
            output.WriteunwrappedState(file_eq_unw, 0, x, y, z, vx, vy, vz)

        # ---------------- Equilibration loop -----------------------------
        logging.info("Equilibration: %d steps", settings.nsteps_eq)
        for i in tqdm(range(settings.nsteps_eq), desc="Eq", leave=False):
            x, y, z, vx, vy, vz = update.update(
                False,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                settings.L,
                settings.N,
                settings.sig,
                settings.delta,
                settings.A,
                settings.m,
                settings.Zprimesqrd,
                settings.lambda_B,
                settings.kappa_D,
                settings.kBT,
                settings.xi,
                settings.delta_t,
                settings.gaus_var,
                seed,
            )
            if write and i % everyN == 0:
                output.WriteTrajectory3d(file_eq, i, x, y, z, settings)
                output.WriteunwrappedState(file_eq_unw, i, x, y, z, vx, vy, vz)

        # ---------------- Production loop ---------------------------------
        logging.info("Production: %d steps", settings.nsteps)
        nbins = int(settings.L / 2 / settings.dr)
        hist = np.zeros(nbins)
        Ngr = 0

        for i in tqdm(range(settings.nsteps), desc="Prod", leave=False):
            x, y, z, vx, vy, vz = update.update(
                False,
                x,
                y,
                z,
                vx,
                vy,
                vz,
                settings.L,
                settings.N,
                settings.sig,
                settings.delta,
                settings.A,
                settings.m,
                settings.Zprimesqrd,
                settings.lambda_B,
                settings.kappa_D,
                settings.kBT,
                settings.xi,
                settings.delta_t,
                settings.gaus_var,
                seed,
            )
            if i % everyN == 0:
                if write:
                    output.WriteTrajectory3d(file_prod, i, x, y, z, settings)
                    output.WriteunwrappedState(file_prod_unw, i, x, y, z, vx, vy, vz)

                hist = update.update_hist(hist, x, y, z, settings.dr, settings.N, settings.L)
                Ngr += 1
                g = update.calcg(Ngr, hist, settings.dr, settings.rho, settings.N)

        # final g(r) if loop ended without update
        if "g" not in locals():
            g = update.calcg(max(Ngr, 1), hist, settings.dr, settings.rho, settings.N)

    return g


# monkey-patch into namespace used further down
# Simulation3 = _Simulation3_patched  # noqa: N816

# ---------------------------------------------------------------------------
# 1.b  Vectorised S(k) helper -----------------------------------------------


def compute_S_of_k_from_gr_vec(g_of_r: np.ndarray, dr: float, rho: float, k_arr: np.ndarray):
    return np.fromiter(
        (compute_S_of_k_from_gr(g_of_r, dr, rho, k) for k in k_arr),
        dtype=float,
        count=len(k_arr),
    )

# ---------------------------------------------------------------------------
# 2. SETTINGS (edit if needed) ----------------------------------------------
# ---------------------------------------------------------------------------

# Cs=10 will be computed seperately
CS_LIST: Sequence[float] = [100, 333, 666, 1000] 
CS_LIST_COMPLETE: Sequence[float] = [10, 100, 333, 666, 1000]
# CS_LIST: Sequence[float] = [100, 333] 
EVERY_N: int = 10

# ---------------------------------------------------------------------------
# 3. Task 3 – run simulations, save g(r) ------------------------------------
# ---------------------------------------------------------------------------

def run_task3(out_dir: Path):
    logging.info("=== Task 3 started ===")
    g_all: list[np.ndarray] = []

    g_of_rs_dir = out_dir / "g_of_rs"       # folder for g(r) stuff
    g_of_rs_dir.mkdir(parents=True, exist_ok=True)

    for Cs in CS_LIST:
        logging.info("Cs %.4g – simulation", Cs)
        settings3.init(Cs)
        g = Simulation3(out_dir, True, f"Task3_Cs{Cs}", EVERY_N, settings3, Cs)
        g_all.append(g)
        
        # logging.info("Cs %.4g – plotting", Cs)
        r_centres = (np.arange(len(g)) + 0.5) * settings3.dr
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r_centres, g)
        ax.set(xlabel=r"$r / \sigma$", ylabel="g(r)", title=f"g(r) – Cs={Cs}")
        fig.tight_layout()
        png = g_of_rs_dir / f"Task3_g(r)_Cs{Cs}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        # logging.info("Saved %s", png.name)
    

    # stacking lsit of 1D‐arrays into one 2D array
    g_matrix = np.vstack(g_all)         # shape = (len(CS_LIST),   n_bins)


    # writing: each row is one g(r)
    out_file = g_of_rs_dir / "Task3_g_all.txt"      
    np.savetxt(out_file, g_matrix)
    logging.info("Saved all g(r) to %s", out_file.name)


    # combined plot ---------------------------------------------------------
    save_combined_gr_plot(out_dir, g_all, CS_LIST)

    logging.info("=== Task 3 Cs=[100, ..., 1000] finished ===")
    return g_all



def run_task3_Cs10(out_dir: Path,
                   g_all: list[np.ndarray],
                   cs_list: list[float]):
    logging.info("=== Task 3 Cs=10 started ===")

    Cs=10

    g_of_rs_dir = out_dir / "g_of_rs"
    g_of_rs_dir.mkdir(parents=True, exist_ok=True)
    g_r_file = g_of_rs_dir / "Task3_g_all.txt"
    # 1) load other g(r) functions from out_dir
    g_matrix = np.loadtxt(g_r_file)

    settings3_Cs10.init(Cs)
    g_Cs10 = Simulation3(out_dir, True, f"Task3_Cs{Cs}", EVERY_N, settings3_Cs10, Cs)

    # put new Cs=10 g(r) as first matrix entry
    g_newmat = np.vstack([g_Cs10, g_matrix])

    # overwrite the old file
    np.savetxt(g_r_file, g_newmat)
    
    logging.info("Cs %.4g – plotting", Cs)
    r_centres = (np.arange(len(g_Cs10)) + 0.5) * settings3_Cs10.dr
    
    # save Cs=10 as well
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_centres, g_Cs10)
    ax.set(xlabel=r"$r / \sigma$", ylabel="g(r)", title=f"g(r) – Cs={Cs}")
    fig.tight_layout()
    png = g_of_rs_dir / f"Task3_g(r)_Cs{Cs}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    logging.info("Saved %s", png.name)

    # now overwrite the combined plot with the new list:
    # prepend the new data
    new_g_all  = [g_Cs10] + g_all
    new_cs_list = [10] + CS_LIST
    
    save_combined_gr_plot(g_of_rs_dir, new_g_all, new_cs_list)
    
    logging.info("=== Task 3 Cs=10 finished ===")
    return new_g_all


def save_combined_gr_plot(out_dir: Path, g_all: list[np.ndarray], cs_list: list[float]):

    r0 = (np.arange(len(g_all[0])) + 0.5) * settings3.dr

    fig, ax = plt.subplots(figsize=(8, 5))
    for Cs, g in zip(cs_list, g_all):
        ax.plot(r0, g, label=f"Cs={Cs}")
    ax.set(xlabel=r"r / σ", ylabel="g(r)")
    ax.legend()
    fig.tight_layout()

    png = out_dir / "Task3_g(r)_all.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    logging.info("Saved combined plot %s", png.name)

# ---------------------------------------------------------------------------
# 4. Task 4 b – compute S(k) & κ_T -----------------------------------------
# ---------------------------------------------------------------------------

def run_task4(out_dir: Path, g_list: list[np.ndarray], cs_list: list):
    logging.info("=== Task 4 b started ===")
    dr = settings3.dr
    L = settings3.L
    rho = getattr(settings3, "rho", None) or settings3.N / L**3
    kBT = getattr(settings3, "kBT", None) or (settings3.kBT)

    k_min = 2 * np.pi / (L / 2)
    k_arr = np.concatenate(([0.0], np.linspace(k_min, 20 * k_min, 400)))
    kappa = []

    sf_dir = out_dir / "structure_factors"          # folder for S(k) stuff
    sf_dir.mkdir(parents=True, exist_ok=True)

    for Cs, g in zip(cs_list, g_list):
        logging.info("Cs %.4g – computing S(k)", Cs)
        S_k = compute_S_of_k_from_gr_vec(g, dr, rho, k_arr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_arr[1:], S_k[1:])
        ax.set(xlabel="k", ylabel="S(k)", title=f"Structure factor – Cs={Cs}")
        fig.tight_layout()
        png = sf_dir / f"structure_factor_Cs{Cs}.png"
        fig.savefig(png, dpi=300)
        plt.close(fig)
        logging.info("Saved %s", png.name)
        if kBT is not None:
            kappa.append((Cs, S_k[0] / (rho * kBT)))

    if kappa:
        logging.info("\nκ_T summary:\n%s", "\n".join(f"Cs={c:8.3g} --> κ_T={k:12.5e}" for c, k in kappa))

    logging.info("=== Task 4 b finished ===")

# ---------------------------------------------------------------------------
# 5. main() – everything together -------------------------------------------
# ---------------------------------------------------------------------------

def main():
    out_dir = create_output_directory()
    setup_logging(out_dir)
    logging.info("Output directory: %s", out_dir)

    # asking to do Cs=10 as well or not
    answer = input("Also run the Cs=10 simulation afterward? [y/N]: ").strip().lower()
    do_cs10 = (answer == "y")

    # run the normal batch (Cs=100,333,…)
    g_list = run_task3(out_dir)

    # only if requested, run Cs=10 at the end
    if do_cs10:
        g_list = run_task3_Cs10(out_dir, g_list, CS_LIST)

    # finally, continue on to task 4 (or whatever comes next)
    run_task4(out_dir, g_list, CS_LIST_COMPLETE)

    logging.info("All done → %s", out_dir)

if __name__ == "__main__":
    main()