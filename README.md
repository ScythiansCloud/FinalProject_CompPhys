# Project 04 – *Simulating Colloidal Flocculation with Langevin Dynamics*

* Task I - Implementing the DLVO potential and the LD integrator
* Task II – validation of the Langevin‐dynamics integrator  
* Task III – MD simulations of five salt concentrations  
* Task IV – post-processing: \(g(r)\), \(S(k)\), coordination numbers,  Figures, helper utilities and the final PDF of the report

## Directories
* `g_r/`    The g(r) files produced by task3_create_g_r
* `output/`  in here there will be created a run_[i] subfolder with the output files of the Task 2 and Task 4 routines
* `settings/` Parameters for the simulations (`settings_task2.py`, `settings_task3.py`, `settings_task3_10.py`)
* `utilities/` helper functions (PBC, MSD, cluster finder, …), initialization and simulation functions

## Dependencies:
numba, numpy, matplotlib, tqdm

## SCRIPTS TO BE RUN
* Task II – `task2.py` – runs the Brownian validation (Task II) and plots MSD + energy  
* Task III – `task3_create_g_r.py` – runs the simulations for the different salt concentrations and writes \(g(r)\) in a txt file in the g_r folder
* Task III/IV - `task3_and_4_plot_g_r_S_k.py` – post-processes the g_r txt files to generate 
* `plotpot.py` – generates the potential curves shown in the report  

## Other scripts
* `task3__.py` – r to run a single salt concetration value's simulation, like Cs=10 because it requires a higher time resolution

## Utilities 

* `force.py`: Computes DLVO + LJ forces, Berendsen thermostat.
* `initialize.py`: Generates the random positions and velocities, assigns Maxwellian velocities,
  removes net momentum and rescales to the target temperature.
* `msd.py`: mean-squared–displacement and plot.
* `output.py`: Writes the OVITO trajectories (`*.xyz`) 
* `update.py`: An Euler–Maruyama integration step, g(r) and S(k) calculation 
* `simulation.py`:the simulation functions for the different tasks (`Simulation2`, `Simulation3`)
  for Task II and Task III runs.
* `utils.py`: auto-indexed `output/run_X/` directory and setup of logging.


