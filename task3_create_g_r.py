from pathlib import Path
import numpy as np

# project imports ------------------------------------------------------------
from utilities.simulation import Simulation3
import settings.settings_task3 as settings_std
import settings.settings_task3_10 as settings_10

# ---------------------------------------------------------------------------
# User‑editable constants
# ---------------------------------------------------------------------------
CS_LIST = [10, 100, 333, 666, 1000]   # salt concentrations to simulate
EVERY_N = 10                          # sampling stride forwarded to Simulation3
GR_DIR = Path("g_r")                 # where to write g_r_Cs*.txt
SIM_DIR = Path("sim_outputs")        # Simulation3 output root (scratch)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def settings_for_cs(Cs: float):
    """Return the correct settings module for this Cs."""
    return settings_10 if int(round(Cs)) == 10 else settings_std


def run_one_cs(Cs: float, sim_dir: Path, every_n: int = EVERY_N):
    """Run Simulation3 for a single Cs and return g(r) array."""
    smod = settings_for_cs(Cs)
    smod.init(Cs)
    name = f"Task3_Cs{int(round(Cs))}"
    print(f"[Task3] Running Simulation3 for Cs={Cs} ...")
    g = Simulation3(sim_dir, True, name, every_n, None, smod, Cs)
    return np.asarray(g, dtype=float).ravel()


def save_gr_txt(g: np.ndarray, Cs: float, gr_dir: Path):
    gr_dir.mkdir(parents=True, exist_ok=True)
    fname = gr_dir / f"g_r_Cs{int(round(Cs))}.txt"
    np.savetxt(fname, g)
    print(f"[Task3] wrote {fname}  (n_bins={len(g)})")
    return fname

# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def main():
    sim_dir = SIM_DIR
    gr_dir = GR_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)
    gr_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Task3 to generate g(r) files for Cs={CS_LIST}")

    for Cs in CS_LIST:
        g = run_one_cs(Cs, sim_dir, every_n=EVERY_N)
        save_gr_txt(g, Cs, gr_dir)

    print("All g(r) files generated.")

if __name__ == "__main__":
    main()
