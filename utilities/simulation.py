import numpy as np
from tqdm import tqdm
from . import initialize
from . import force
from . import update
import settings.settings_task2 as settings
# import settings_task3 as settings
from . import output
import logging

def Simulation(outdir, write, Traj_name, everyN):

    # random seed for reproducibility
    np.random.seed(settings.random_seed)
    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, settings.random_seed)
    x,y,z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
                                      settings.delta, settings.A, settings.m, settings.Zprimesqrd,
                                      settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
                                      settings.delta_t, settings.gaus_var, settings.random_seed)
    

    # rescale velocities should not be needed here, as we do this already in initialize 
    # T_curr = initialize.temperature(vx, vy, vz)
    # vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.kBT, T_curr)
    
    print(f'vx, vy, vz = {vx[0], vy[0], vz[0]}')

    if write:
        fileoutput_eq = open(outdir / (Traj_name + str(everyN) + '_nsteps_' + str(settings.nsteps)), "w")
        output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 

    for i in tqdm(range(settings.nsteps)):
        x,y,z, vx, vy, vz = update.update(False, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
                                        settings.delta, settings.A, settings.m, settings.Zprimesqrd,
                                        settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
                                        settings.delta_t, settings.gaus_var, settings.random_seed)
        
        # # Temp rescaling
        kBT_current, Kin = force.temperature(vx, vy, vz, settings)
        force.berendsen_thermostat(vx, vy, vz, kBT_current, 1, delta_t=settings.delta_t, tau_berendsen=settings.tau_berendsen)
        
        # save shit every n
        if i % everyN == 0:

            if write:
                output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 


def Simulation2(outdir, write, Traj_name, everyN): # forces turned off

    # random seed for reproducibility
    np.random.seed(settings.random_seed)
    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings.Cs, settings.random_seed)
    x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
                                      settings.delta, settings.A, settings.m, settings.Zprimesqrd,
                                      settings.lambda_B, settings.kappa_D, settings.kBT, settings.xi,
                                      settings.delta_t, settings.gaus_var, settings.random_seed)
    

    # rescale velocities should not be needed here, as we do this already in initialize 
    # T_curr = initialize.temperature(vx, vy, vz)
    # vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.kBT, T_curr)
    
    print(f'vx, vy, vz = {vx[0], vy[0], vz[0]}')

    if write:
        fileoutput_eq = open(outdir / (Traj_name + str(everyN) + '_nsteps_' + str(settings.nsteps)), "w")
        fileoutput_eq_unwrapped = open(outdir / (Traj_name+ 'unwrapped'+ str(everyN) + '_nsteps_' + str(settings.nsteps)), "w")
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
                logging.info(force.temperature(vx,vy,vz,settings=settings))
                output.WriteTrajectory3d(fileoutput_eq, i,x,y,z, settings)
                output.WriteunwrappedState(fileoutput_eq_unwrapped,i,x,y,z,vx,vy,vz)

