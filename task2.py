#!/usr/bin/env python3
'''
Task 2

1. Make an output folder + logger
2. Run a quick (non-interacting) Langevin-dynamics simulation
3. Read the plain-text trajectory as it’s written
4. Plot ⟨E_kin⟩  and MSD
'''

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utilities.simulation import Simulation2
from utilities.msd import compute_msd, plot_msd
from utilities import utils
import settings.settings_task2 as settings


# ─── tweak-here ──────────────────────────────────────────────────────────────
TRAJ_NAME = "Task2"   # file prefix
EVERY_N   = 10        # store a frame every N MD steps
SAVE_FIG  = True      # write the PNG at the end?
LOG_EVERY = 2_000     # print read progress every X frames
# ─────────────────────────────────────────────────────────────────────────────


def load_unwrapped_state(
    filepath: Path,
    *,
    mass: float | None = None,
    n_particles: int,
    expected_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    '''Read the trajectory produced by the simulation2 function

    Returns:
        times, positions, velocities, KE (optional)
    '''
    logging.info('Reading trajectory → %s', filepath)

    n_numbers = 1 + 6 * n_particles  # t + (x y z vx vy vz) * N
    logging.debug('Need %d numbers per frame', n_numbers)

    def tokens(line: str) -> list[str]:
        # strip fancy formatting LAMMPS-style output might add
        return (
            line.replace('[', ' ')
                .replace(']', ' ')
                .replace(',', ' ')
                .replace('x:', ' ').replace('y:', ' ').replace('z:', ' ')
                .replace('vx:', ' ').replace('vy:', ' ').replace('vz:', ' ')
                .split()
        )

    times, pos_chunks, vel_chunks, stash = [], [], [], []
    frames = 0

    with open(filepath) as fh:
        for raw in fh:
            if not raw.strip():
                continue
            stash.extend(tokens(raw))

            # pull out complete frames as soon as we can
            while len(stash) >= n_numbers:
                frame, stash = stash[:n_numbers], stash[n_numbers:]
                try:
                    data = np.asarray(frame, float)
                except ValueError:
                    logging.warning('Bad frame starting with %s … skipped', frame[:6])
                    continue

                times.append(data[0])
                pv = data[1:].reshape(n_particles, 6)
                pos_chunks.append(pv[:, :3])
                vel_chunks.append(pv[:, 3:])
                frames += 1

                if frames % LOG_EVERY == 0:
                    pct = 100 * frames / expected_frames
                    logging.info('Read %d / %d frames (%.1f %%)', frames, expected_frames, pct)

    if stash:
        logging.warning('Dropped %d stray numbers at EOF', len(stash))
    if not frames:
        raise RuntimeError('No usable frames – aborting.')

    # stack into tidy arrays
    positions  = np.stack(pos_chunks)
    velocities = np.stack(vel_chunks)
    times      = np.asarray(times)

    ke = None
    if mass is not None:
        ke = 0.5 * mass * (velocities**2).sum(-1).mean(-1)  # ⟨v²⟩ per particle

    return times, positions, velocities, ke


# ─── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1) folders + logger
    outdir = utils.create_output_directory()
    utils.setup_logging(outdir)
    logging.info('=== Task 2 started ===')

    # 2) run the tiny LD sim
    settings.init(1)  # salt conc. irrelevant here
    Simulation2(outdir, write=True, Traj_name=TRAJ_NAME, everyN=EVERY_N)

    # 3) crunch the trajectory
    traj_file = outdir / f'{TRAJ_NAME}unwrapped{EVERY_N}_nsteps_{settings.nsteps}'
    n_frames  = settings.nsteps // EVERY_N + 1  # +1 for t=0

    times, pos, vel, ke = load_unwrapped_state(
        traj_file,
        mass=settings.m,
        n_particles=settings.N,
        expected_frames=n_frames,
    )

    dt_snap  = EVERY_N * settings.delta_t
    lags, msd = compute_msd(pos, max_lag=len(times)//2)

    # 4) plots
    fig, (ax_ke, ax_msd) = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

    # Kinetic energy
    ax_ke.plot(times * settings.delta_t, ke)
    ax_ke.set_xlabel(r'Time $t\,[\tau_{LD}]$')
    ax_ke.set_ylabel(r'$\langle E_{kin}\rangle$ / particle')
    ax_ke.set_title('Kinetic Energy')

    # MSD
    plot_msd(ax_msd, lags, msd, dt=dt_snap)
    ax_msd.set_xlabel(r'Time $t\,[\tau_{LD}]$')
    ax_msd.set_ylabel(r'$\langle r^{2}(t)\rangle$')

    # quick linear fit of the tail (last 30 %)
    start = int(0.7 * len(lags))
    slope, intercept = np.polyfit(lags[start:] * dt_snap, msd[start:], 1)
    D_est = slope / 6
    ax_msd.plot(lags * dt_snap,
                slope * lags * dt_snap + intercept,
                '--', label=f'D ≈ {D_est:.3g}')
    ax_msd.legend()

    if SAVE_FIG:
        fig.savefig(outdir / 'energy_msd.png', dpi=300)
        logging.info('Figure saved → %s', outdir / 'energy_msd.png')

    logging.info('=== Task 2 done – %d frames analysed ===', len(times))


if __name__ == '__main__':
    main()
