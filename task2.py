#!/usr/bin/env python3
'''
Task 2

This file first creates a now output folder, and then runs an interaction less Langevin MD Simulation. 
Thereafter it calculates the MSD and kinetic energy. and plots these + balisitc and dissipative fit.
'''

#from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utilities.simulation import Simulation2
from utilities.msd import compute_msd, plot_msd
from utilities import utils
import settings.settings_task2 as settings
from scipy.optimize import curve_fit


# settings
settings.init(10)
traj_name = "Task2"  
nsave   = settings.nsave
SAVE_FIG  = True   
log_every = 2000 

############################
def load_unwrapped_state(filepath, mass, N, ):
    data = np.loadtxt(filepath) # type: ignore

    times = data[:, 0]
    x  = data[:, 1          : 1 + N] # hardcoded for 3D 
    y  = data[:, 1 + N   : 1 + 2*N]
    z  = data[:, 1 + 2*N  : 1 + 3*N]
    vx = data[:, 1 + 3*N  : 1 + 4*N]
    vy = data[:, 1 + 4*N : 1 + 5*N]
    vz = data[:, 1 + 5*N  : 1 + 6*N]

    positions  = np.stack((x, y, z),  axis=-1)           # (f, N, 3)
    velocities = np.stack((vx, vy, vz), axis=-1)         # (f, N, 3)

    ke = 0.5 * mass * (velocities**2).sum(-1)  
    ke = ke.mean(-1)                    

    return times, positions, velocities, ke


def parabola(t, a):#, b):
    return a * t**2# + b

def lin(t, m, c):
    return m * t + c



#################################
def main():
    # init stuff
    outdir = utils.create_output_directory()
    utils.setup_logging(outdir)
    logging.info('Task 2 started')

    # run simulation
    settings.init(1)  # concentration irrelevant
    Simulation2(outdir, write=True, Traj_name=traj_name, everyN=nsave, random_seed=settings.random_seed, settings=settings)

    #calculate msd
    traj_file = outdir / f'{traj_name}unwrapped{nsave}_nsteps_{settings.nsteps}' # using 'path' type instead of strings


    times, pos, vel, ke = load_unwrapped_state(traj_file, settings.m, settings.N)

    dt_snap  = nsave * settings.delta_t
    lags, msd = compute_msd(pos, max_lag=len(times)//2)

    #plot everything
    tlags = lags * dt_snap
    plt.figure(dpi= 600)
    plt.title('Kinetic Energy')
    plt.xlabel(r'$t\,[\tau_{LD}]$')
    plt.ylabel(r'$\langle E_{kin}\rangle$ / particle')
    plt.plot(times * settings.delta_t, ke, color = 'black', label=r'K(t)')
    plt.legend()
    if SAVE_FIG:
        plt.savefig(outdir / 'energy.png')
        logging.info('Figure saved → %s', outdir / 'energya.png')

    plt.figure(dpi=600)
    plt.title('Mean-square-distance')
    plt.xlabel(r'Lag-time $[\tau_{LD}]$')
    plt.ylabel(r'$\langle r^{2}(t)\rangle$')
    plt.plot(tlags,msd,color= 'black', label= 'Mean-Square-Distance')
    # fit balistic, and diffusive regime
    popt, _ = curve_fit(lin, lags[int(0.7 * len(lags)):] * dt_snap, msd[int(0.7 * len(lags)):])
    m, c = popt[0], popt[1]
    D = m / 6

    plt.plot(tlags[int(0.07 * len(lags)):], lin(tlags[int(0.07 * len(lags)):], m, c), '--', color = 'red', label=f'Diffusive Regime; D = {round(D,3)}')


    popt2, _ = curve_fit(parabola,lags[:int(0.001 * len(lags))] * dt_snap, msd[:int(0.001 * len(lags)):])
    a = popt2[0]#, popt2[1]
    plt.plot(tlags[:int(0.05 * len(lags))], parabola(tlags[:int(0.05 * len(lags))],a)[:int(0.05 * len(lags))], '--', color= 'blue', label=f'Ballistic regime')


    plt.legend()

    plt.xscale('log')
    plt.yscale('log')
    if SAVE_FIG:
        plt.savefig(outdir / 'msd.png', dpi=300)
        logging.info('Figure saved → %s', outdir / 'msd.png')

    logging.info('Fit begin diff: '+str(tlags[int(0.7 * len(lags))]))
    logging.info('Fit ende ball: '+ str( tlags[int(0.001 * len(lags))]))

    


    #     t = lags * dt
    # line, = ax.plot(t, msd, **kwargs)
    # ax.set_xlabel(r"Time $t\,[\tau_{\mathrm{LD}}]$")
    # ax.set_ylabel(r"$\langle r^{2}(t) \rangle\,[\sigma^{2}]$")
    # ax.set_title("Mean‑Squared Displacement")
    # return line



    # fig, (ax_ke, ax_msd) = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

    # # Kinetic energy
    # ax_ke.plot(times * settings.delta_t, ke)
    # ax_ke.set_xlabel(r'Time $t\,[\tau_{LD}]$')
    # ax_ke.set_ylabel(r'$\langle E_{kin}\rangle$ / particle')
    # ax_ke.set_title('Kinetic Energy')

    # # MSD
    # plot_msd(ax_msd, lags, msd, dt=dt_snap)
    # ax_msd.set_xlabel(r'Time $t\,[\tau_{LD}]$')
    # ax_msd.set_ylabel(r'$\langle r^{2}(t)\rangle$')

    # # quick linear fit of the tail (last 30 %)
    # start = int(0.7 * len(lags))
    # slope, intercept = np.polyfit(lags[start:] * dt_snap, msd[start:], 1)
    # D_est = slope / 6
    # ax_msd.plot(lags * dt_snap,
    #             slope * lags * dt_snap + intercept,
    #             '--', label=f'D ≈ {D_est:.3g}')
    # ax_msd.set_xscale('log')
    # ax_msd.set_yscale('log')
    # ax_msd.legend()

    # if SAVE_FIG:
    #     fig.savefig(outdir / 'energy_msd.png', dpi=300)
    #     logging.info('Figure saved → %s', outdir / 'energy_msd.png')

    logging.info('=== Task 2 done – %d frames analysed ===', len(times))


if __name__ == '__main__':
    main()


















# 'def load_unwrapped_state(
#     filepath: Path,
#     *,
#     mass: float | None = None,
#     n_particles: int,
#     expected_frames: int,
# ):
#     data = np.loadtxt(filepath)

#     times = data[0]
#     x = data[:, 1:settings.N+1]
#     y = data[:, 1+settings.N:settings.N*2+1]
#     z = data[:, 1+settings.N*2:1+settings.N*3]
#     vx = data[:, 1+settings.N*3:1+settings.N*4]
#     vy = data[:, 1+settings.N*4:1+settings.N*5]
#     vz = data[:, 1+settings.N*5:1+settings.N*6]

#     positions = np.stack((x, y, z), axis=-1) # (t,N,3) weis nicht ob das die shape ist mit der du garbeitet hast...
#     velocities = np.stack((vx, vy, vz), axis=-1)
#     ke =( vx**2+vy**2+vz**2)*1/2 # mass = 1 spuckt irgendwie sonst einen fehler aus

#     return times, positions, velocities, ke'

# def load_unwrapped_state(
#     filepath: Path,
#     *,
#     mass: float | None = None,
#     n_particles: int,
#     expected_frames: int,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
#     '''Read the trajectory produced by the simulation2 function

#     Returns:
#         times, positions, velocities, KE (optional)
#     '''
#     logging.info('Reading trajectory → %s', filepath)

#     n_numbers = 1 + 6 * n_particles  # t + (x y z vx vy vz) * N
#     logging.debug('Need %d numbers per frame', n_numbers)

#     def tokens(line: str) -> list[str]:
#         # strip fancy formatting LAMMPS-style output might add
#         return (
#             line.replace('[', ' ')
#                 .replace(']', ' ')
#                 .replace(',', ' ')
#                 .replace('x:', ' ').replace('y:', ' ').replace('z:', ' ')
#                 .replace('vx:', ' ').replace('vy:', ' ').replace('vz:', ' ')
#                 .split()
#         )

#     times, pos_chunks, vel_chunks, stash = [], [], [], []
#     frames = 0

#     with open(filepath) as fh:
#         for raw in fh:
#             if not raw.strip():
#                 continue
#             stash.extend(tokens(raw))

#             # pull out complete frames as soon as we can
#             while len(stash) >= n_numbers:
#                 frame, stash = stash[:n_numbers], stash[n_numbers:]
#                 try:
#                     data = np.asarray(frame, float)
#                 except ValueError:
#                     logging.warning('Bad frame starting with %s … skipped', frame[:6])
#                     continue

#                 times.append(data[0])
#                 pv = data[1:].reshape(n_particles, 6)
#                 pos_chunks.append(pv[:, :3])
#                 vel_chunks.append(pv[:, 3:])
#                 frames += 1

#                 if frames % log_every == 0:
#                     pct = 100 * frames / expected_frames
#                     logging.info('Read %d / %d frames (%.1f %%)', frames, expected_frames, pct)

#     if stash:
#         logging.warning('Dropped %d stray numbers at EOF', len(stash))
#     if not frames:
#         raise RuntimeError('No usable frames – aborting.')

#     # stack into tidy arrays
#     positions  = np.stack(pos_chunks)
#     velocities = np.stack(vel_chunks)
#     times      = np.asarray(times)

#     ke = None
#     if mass is not None:
#         ke = 0.5 * mass * (velocities**2).sum(-1).mean(-1)  # ⟨v²⟩ per particle

#     return times, positions, velocities, ke

#def load_unwrapped_state(
#     filepath: Path,
#     *,
#     mass: float | None,
#     n_particles: int,
#     expected_frames: int,
# ):
#     data = np.loadtxt(filepath)

#     # Basic sanity check
#     if data.shape[0] != expected_frames:
#         logging.warning(
#             "Expected %d frames but found %d in %s",
#             expected_frames, data.shape[0], filepath,
#         )

#     # 1) split the big table --------------------------------------------------
#     times = data[:, 0]                                   # (f,)

#     blk = n_particles
#     x  = data[:, 1          : 1 + blk]
#     y  = data[:, 1 + blk    : 1 + 2*blk]
#     z  = data[:, 1 + 2*blk  : 1 + 3*blk]
#     vx = data[:, 1 + 3*blk  : 1 + 4*blk]
#     vy = data[:, 1 + 4*blk  : 1 + 5*blk]
#     vz = data[:, 1 + 5*blk  : 1 + 6*blk]

#     positions  = np.stack((x, y, z),  axis=-1)           # (f, N, 3)
#     velocities = np.stack((vx, vy, vz), axis=-1)         # (f, N, 3)

#     # 2) kinetic energy per particle, then ⟨⋯⟩ over all particles -------------
#     m = 1.0 if mass is None else mass
#     ke = 0.5 * m * (velocities**2).sum(-1)   # (f, N) – v² per particle
#     ke = ke.mean(-1)                         # (f,)   – ⟨E_kin⟩

#     return times, positions, velocities, ke
