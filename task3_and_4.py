from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Imports and helper functions
# ---------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np

# Project-specific imports ----------------------------------------------------
from utilities.simulation import Simulation3
from utilities.update import compute_S_of_k_from_gr  # scalar routine
import settings.settings_task3 as settings3
import settings.settings_task3_10 as settings3_Cs10
from utilities.utils import create_output_directory, setup_logging

# ---------------------------------------------------------------------------
# 1. SETTINGS  ----------------------------------------------
# ---------------------------------------------------------------------------

# Cs=10 will be computed seperately
CS_LIST: Sequence[float] = [100, 333, 666, 1000] 
CS_LIST_COMPLETE: Sequence[float] = [10, 100, 333, 666, 1000]
# CS_LIST: Sequence[float] = [100, 333] 
EVERY_N: int = 10

# ---------------------------------------------------------------------------
# 2. Task 3 – run simulations, save g(r) ------------------------------------
# ---------------------------------------------------------------------------

def run_task3(out_dir: Path):
    logging.info("=== Task 3 started ===")
    g_all: list[np.ndarray] = []

    g_of_rs_dir = out_dir / "g_of_rs"       # folder for g(r) stuff
    g_of_rs_dir.mkdir(parents=True, exist_ok=True)

    for Cs in CS_LIST:
        logging.info("Cs %.4g – simulation", Cs)
        settings3.init(Cs)
        g = Simulation3(out_dir, True, f"Task3_Cs{Cs}", EVERY_N, None, settings3, Cs)
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

# ---------------------------------------------------------------------------
# 3. Task 3 Cs=10 – run simulation, save g(r) --------------------------------
# ---------------------------------------------------------------------------

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
    g_Cs10 = Simulation3(out_dir, True, f"Task3_Cs{Cs}", EVERY_N, None, settings3_Cs10, Cs)

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

# ---------------------------------------------------------------------------
# 4. Task 3 g(r) combined plot – save to file --------------------------------  
# ---------------------------------------------------------------------------

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
# 5. Task 4 b – compute S(k) & κ_T -----------------------------------------
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
        S_k = compute_S_of_k_from_gr(g, dr, rho, k_arr)
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
# 6. main() – everything together -------------------------------------------
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