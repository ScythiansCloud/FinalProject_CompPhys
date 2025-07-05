#!/usr/bin/env python3
"""Complete workflow for *Task 2* – validation of the Langevin–Dynamics (LD)
integration scheme by analysing the mean‑squared displacement (MSD) and the
kinetic energy of an ideal gas of colloids.

This script performs **all** required steps in one go:

1. Creates a timestamped output directory and initialises logging.
2. Runs an LD simulation with *all inter‑particle potentials switched off*
   using :pyfunc:`utilities.simulation.Simulation2`.  The simulation writes
   wrapped and unwrapped trajectories to text files in the output directory.
3. Parses the unwrapped trajectory and computes
   * the instantaneous kinetic energy per particle, and
   * the mean‑squared displacement (MSD) according to Eq. (4) of the
     assignment sheet using :pyfunc:`utilities.msd.compute_msd`.
4. Produces a single PNG figure containing both diagnostics and saves it to
   the output directory.

Adjust *settings/settings_task2.py* to change physical parameters, and tweak
*everyN* below to alter the snapshot saving frequency.  The script can be run
from the project root via:

```bash
python main_task2.py
```
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utilities.simulation import Simulation2
from utilities import utils

import settings.settings_task2 as settings
from utilities.msd import compute_msd, plot_msd


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

TRAJ_NAME = "Task2"          # prefix for output trajectory files
EVERY_N = 10                 # write unwrapped state every N integration steps
SAVE_FIG = True


# ----------------------------------------------------------------------------
# Helper – parse the ASCII trajectory written by WriteunwrappedState
# ----------------------------------------------------------------------------

def load_unwrapped_state(filepath: Path, mass: float | None = None):
    """Read the unwrapped trajectory produced by *WriteunwrappedState*.

    The writer sometimes encloses numeric arrays in square brackets and/or
    separates values with commas – e.g. ``[0.24326, 1.02, ...]``.  This helper
    therefore *cleans* every line by stripping ``[]`` and ``,`` before token
    parsing, making the reader tolerant to a variety of formatting issues.
    Any line that still does not match the expected column count is skipped
    with a warning.
    """

    logging.info("Reading unwrapped state from %s", filepath)

    def clean_tokens_from_line(line: str) -> list[str]:
        """Remove brackets & commas, then split into tokens."""
        return (
            line.replace("[", " ")
            .replace("]", " ")
            .replace(",", " ")
            .split()
        )

    with open(filepath, "r") as fh:
        # Find first non-empty, well‑formed line to infer n_particles
        for first_raw in fh:
            if first_raw.strip():
                tokens0 = clean_tokens_from_line(first_raw)
                if tokens0:
                    break
        else:
            raise RuntimeError("Trajectory file appears to be empty.")

        n_cols0 = len(tokens0)
        if (n_cols0 - 1) % 6 != 0 or n_cols0 < 7:
            raise ValueError(
                f"Cannot deduce particle number from first data line ({n_cols0} columns)."
            )
        n_particles = (n_cols0 - 1) // 6
        expected_cols = 1 + 6 * n_particles
        logging.debug("Detected %d particles from trajectory file", n_particles)

        # Buffers
        times: list[float] = []
        positions_lst: list[np.ndarray] = []
        velocities_lst: list[np.ndarray] = []

        def process(tokens: list[str]):
            data = np.array(tokens, dtype=float)
            times.append(data[0])
            pv = data[1:].reshape(n_particles, 6)
            positions_lst.append(pv[:, :3])
            velocities_lst.append(pv[:, 3:])

        # First line processed (only if token count matches expectation)
        if len(tokens0) == expected_cols:
            process(tokens0)
        else:
            logging.warning("Skipping first malformed line with %d tokens (expected %d)", len(tokens0), expected_cols)

        # Remaining lines
        for raw in fh:
            if not raw.strip():
                continue
            tokens = clean_tokens_from_line(raw)
            if len(tokens) != expected_cols:
                logging.warning(
                    "Skipping line with %d tokens (expected %d)", len(tokens), expected_cols
                )
                continue
            process(tokens)

    if not times:
        raise RuntimeError("No valid trajectory lines read – aborting analysis.")

    positions = np.stack(positions_lst, axis=0)
    velocities = np.stack(velocities_lst, axis=0)
    times = np.asarray(times, dtype=float)

    if n_particles is None:
        # fall back to auto-detect (current behaviour)
        infer n_particles from first line
    else:
        expected_cols = 1 + 6 * n_particles

    ke_per_frame = None
    if mass is not None:
        ke_per_frame = 0.5 * mass * (velocities ** 2).sum(axis=-1).mean(axis=-1)

    return times, positions, velocities, ke_per_frame


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------

def main():
    # Create output directory & logging as per project conventions
    output_dir = utils.create_output_directory()
    utils.setup_logging(output_dir)

    logging.info("=== Task 2 run started ===")

    # ---------------------------------------------------------------------
    # 1) Run the LD simulation with interactions off
    # ---------------------------------------------------------------------

    settings.init(1)  # salt conc. arbitrary – potentials are off inside Simulation2

    Simulation2(output_dir, write=True, Traj_name=TRAJ_NAME, everyN=EVERY_N)

    # ---------------------------------------------------------------------
    # 2) Analysis – load trajectory, compute KE and MSD
    # ---------------------------------------------------------------------

    traj_file = (
        output_dir
        / f"{TRAJ_NAME}unwrapped{EVERY_N}_nsteps_{settings.nsteps}"
    )
    if not traj_file.exists():
        raise FileNotFoundError(traj_file)

    times, positions, velocities, ke_per_frame = load_unwrapped_state(
    traj_file, mass=settings.m, n_particles=settings.N
    )

    # The simulation saves every EVERY_N integration steps => physical Δt between
    # *saved* frames is EVERY_N * settings.delta_t.
    dt_snapshot = EVERY_N * settings.delta_t

    # Compute MSD using the helper from utilities.msd
    max_lag = len(times) // 2  # ensure consistency with assignment text
    lags, msd = compute_msd(positions, max_lag=max_lag)

    # ---------------------------------------------------------------------
    # 3) Plot results
    # ---------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

    # Kinetic energy per particle vs. simulation time
    if ke_per_frame is not None:
        ax1.plot(times * settings.delta_t, ke_per_frame)
        ax1.set_xlabel("Time t")
        ax1.set_ylabel(r"$\langle E_\text{kin} \rangle$ per particle")
        ax1.set_title("Kinetic Energy vs. Time")

    # Mean‑squared displacement
    plot_msd(ax2, lags, msd, dt=dt_snapshot)

    # Slope estimate for long‑time diffusion (fit last third of the MSD curve)
    fit_from = int(0.7 * len(lags))
    coeffs = np.polyfit(lags[fit_from:] * dt_snapshot, msd[fit_from:], 1)
    D_est = coeffs[0] / 6.0  # MSD ≈ 6 D t in 3‑D
    ax2.plot(
        lags * dt_snapshot,
        coeffs[0] * lags * dt_snapshot + coeffs[1],
        ls="--",
        label=fr"fit ⇒ D ≈ {D_est:.3g}",
    )
    ax2.legend()

    if SAVE_FIG:
        figfile = output_dir / "energy_msd.png"
        fig.savefig(figfile, dpi=300)
        logging.info("Figure saved to %s", figfile)

    logging.info("=== Task 2 completed successfully ===")


if __name__ == "__main__":
    main()
