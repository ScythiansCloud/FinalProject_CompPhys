import numpy as np
from tqdm import tqdm
from . import initialize
from . import force
from . import update
#import settings.settings_task2 as settings
#import settings.settings_task3 as settings3
from . import output
import logging

# def Simulation(outdir, write, Traj_name, everyN):

#     # random seed for reproducibility
#     np.random.seed(settings.random_seed)
#     x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, settings.random_seed, settings)
#     x,y,z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
#                                       settings.delta, settings.A, settings.m, settings.Zprimesqrd,
#                                       settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
#                                       settings.delta_t, settings.gaus_var, settings.random_seed)
    

#     # rescale velocities should not be needed here, as we do this already in initialize 
#     # T_curr = initialize.temperature(vx, vy, vz)
#     # vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.kBT, T_curr)
    
#     print(f'vx, vy, vz = {vx[0], vy[0], vz[0]}')

#     if write:
#         fileoutput_eq = open(outdir / (Traj_name + str(everyN) + '_nsteps_' + str(settings.nsteps)), "w")
#         output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 
#         # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 

#     for i in tqdm(range(settings.nsteps)):
#         x,y,z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
#                                         settings.delta, settings.A, settings.m, settings.Zprimesqrd,
#                                         settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
#                                         settings.delta_t, settings.gaus_var, settings.random_seed)
        
#         # # Temp rescaling
#         kBT_current, Kin = force.temperature(vx, vy, vz, settings)
#         force.berendsen_thermostat(vx, vy, vz, kBT_current, 1, delta_t=settings.delta_t, tau_berendsen=settings.tau_berendsen)
        
#         # save shit every n
#         if i % everyN == 0:

#             if write:
#                 output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 


def Simulation2(outdir, write, Traj_name, everyN, random_seed, settings): # forces turned off
    import settings.settings_task2 as settings
    # random seed for reproducibility
    np.random.seed(settings.random_seed)
    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, settings.random_seed, settings)
    x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
                                      settings.delta, settings.A, settings.m, settings.Zprimesqrd,
                                      settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
                                      settings.delta_t, settings.gaus_var, settings.random_seed)
    

    # rescale velocities should not be needed here, as we do this already in initialize 
    # T_curr = initialize.temperature(vx, vy, vz)
    # vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.kBT, T_curr)
    
    print(f'vx, vy, vz = {vx[0], vy[0], vz[0]}')

    if write:
        fileoutput_eq = open(outdir / (Traj_name + "_eq" + '_everyN' + str(everyN) + '_nsteps' + str(settings.nsteps)), "w")
        fileoutput_eq_unwrapped = open(outdir / (Traj_name+ '_unwrapped'+ '_everyN' + str(everyN) + '_nsteps' + str(settings.nsteps)), "w")
        output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 
        output.WriteunwrappedState(fileoutput_eq_unwrapped,0,x,y,z,vx,vy,vz)
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 

    for i in tqdm(range(settings.nsteps)):
        x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
                                        settings.delta, settings.A, settings.m, settings.Zprimesqrd,
                                        settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
                                        settings.delta_t, settings.gaus_var, settings.random_seed)
        
        # save shit every n
        if i % everyN == 0:
            if write:
                # logging.info(force.temperature(vx,vy,vz,settings=settings))
                output.WriteTrajectory3d(fileoutput_eq, i,x,y,z, settings)
                output.WriteunwrappedState(fileoutput_eq_unwrapped,i,x,y,z,vx,vy,vz)



def Simulation3(outdir, write, Traj_name, everyN, random_seed, settings, Csi):  # forces turned off
    settings.init(Csi)

    # ------------------------------------------------------------------
    # random seed for reproducibility
    # ------------------------------------------------------------------
    seed = settings.random_seed if random_seed is None else random_seed       # <-- changed
    np.random.seed(seed)                                                      # <-- changed

    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, seed, settings)  # <-- changed
    x, y, z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L,
                                        settings.N, settings.sig, settings.delta,
                                        settings.A, settings.m, settings.Zprimesqrd,
                                        settings.lambda_B, settings.kappa_D,
                                        settings.kBT, settings.xi, settings.delta_t,
                                        settings.gaus_var, seed)                     # <-- changed
    # ------------------------------------------------------------------

    # open equilibration files
    if write:
        fileoutput_eq = open(outdir / (Traj_name + "_eq_" + str(everyN) + '_nsteps' + str(settings.nsteps)), "w")
        fileoutput_eq_unwrapped = open(outdir / (Traj_name+ '_unwrapped'+ str(everyN) + '_nsteps_' + str(settings.nsteps)), "w")
        output.WriteTrajectory3d(fileoutput_eq, 0, x, y, z, settings) 
        output.WriteunwrappedState(fileoutput_eq_unwrapped, 0, x, y, z, vx, vy, vz)
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 
    
    # equilibration run
    logging.info(f"Starting with equilibration run of {settings.nsteps_eq} steps.")
    for i in tqdm(range(settings.nsteps_eq)):
        x, y, z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L,
                                            settings.N, settings.sig, settings.delta,
                                            settings.A, settings.m, settings.Zprimesqrd,
                                            settings.lambda_B, settings.kappa_D,
                                            settings.kBT, settings.xi, settings.delta_t,
                                            settings.gaus_var, seed)                 # <-- changed
        
        # save shit every n
        if i % everyN == 0:
            if write:
                # logging.info(force.temperature(vx,vy,vz,settings=settings))
                output.WriteTrajectory3d(fileoutput_eq, i, x, y, z, settings)
                output.WriteunwrappedState(fileoutput_eq_unwrapped, i, x, y, z, vx, vy, vz)
    

    # open PRODUCTION files
    if write:
        fileoutput_prod = open(outdir / (Traj_name + '_everyN' + str(everyN) + '_nsteps' + str(settings.nsteps)), "w")
        fileoutput_prod_unwrapped = open(outdir / (Traj_name+ '_unwrapped'+ '_everyN' + str(everyN) + '_nsteps' + str(settings.nsteps)), "w")
        output.WriteTrajectory3d(fileoutput_prod, 0, x, y, z, settings) 
        output.WriteunwrappedState(fileoutput_prod_unwrapped, 0, x, y, z, vx, vy, vz)
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 
    
    # production run
    logging.info(f"Starting with production run of {settings.nsteps} steps.")
    Ngr = 0
    nbins = int(settings.L / 2 / settings.dr)
    hist = np.zeros(nbins) 
    for i in tqdm(range(settings.nsteps)):
        x, y, z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L,
                                            settings.N, settings.sig, settings.delta,
                                            settings.A, settings.m, settings.Zprimesqrd,
                                            settings.lambda_B, settings.kappa_D,
                                            settings.kBT, settings.xi, settings.delta_t,
                                            settings.gaus_var, seed)                 # <-- changed
        
        # save shit every n
        if i % everyN == 0:
            if write:
                # logging.info(force.temperature(vx,vy,vz,settings=settings))
                output.WriteTrajectory3d(fileoutput_prod, i, x, y, z, settings)
                output.WriteunwrappedState(fileoutput_prod_unwrapped, i, x, y, z, vx, vy, vz)

            # ----------------------------------------------------------
            # histogram/g(r) update only when we actually recorded a frame
            # ----------------------------------------------------------
            hist = update.update_hist(hist, x, y, z, settings.dr, settings.N, settings.L)
            Ngr += 1  # another position
            g = update.calcg(Ngr, hist, settings.dr, settings.rho, settings.N)       # <-- moved inside block
    # ------------------------------------------------------------------

    # make sure we return a g(r) even if no frame satisfied i % everyN == 0
    if 'g' not in locals():                                                    # <-- added
        g = update.calcg(max(Ngr, 1), hist, settings.dr, settings.rho, settings.N)   # <-- added
                    
    return g
