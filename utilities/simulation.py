import numpy as np
from tqdm import tqdm
from . import initialize
from . import force
from . import update
import settings.settings_task2 as settings
import settings.settings_task3 as settings3
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
                # logging.info(force.temperature(vx,vy,vz,settings=settings))
                output.WriteTrajectory3d(fileoutput_eq, i,x,y,z, settings)
                output.WriteunwrappedState(fileoutput_eq_unwrapped,i,x,y,z,vx,vy,vz)



def Simulation3(outdir, write, Traj_name, everyN): # forces turned off

    # random seed for reproducibility
    np.random.seed(settings3.random_seed)
    x, y, z, vx, vy, vz = initialize.InitializeAtoms(settings3.Cs, settings3.random_seed)
    x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings3.L, settings3.N, settings3.sig,
                                      settings3.delta, settings3.A, settings3.m, settings3.Zprimesqrd,
                                      settings3.lambda_B, settings3.kappa_D, settings3.kBT, settings3.xi,
                                      settings3.delta_t, settings3.gaus_var, settings3.random_seed)
    

    if write:
        fileoutput_eq = open(outdir / (Traj_name + str(everyN) + '_nsteps_' + str(settings3.nsteps)), "w")
        fileoutput_eq_unwrapped = open(outdir / (Traj_name+ 'unwrapped'+ str(everyN) + '_nsteps_' + str(settings3.nsteps)), "w")
        output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings3) 
        output.WriteunwrappedState(fileoutput_eq_unwrapped,0,x,y,z,vx,vy,vz)
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 
    
    Ngr = 0
    nbins = int(settings3.l/2 / settings3.dr)
    hist = np.zeros(nbins) 
    for i in tqdm(range(settings3.nsteps)):
        x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings3.L, settings3.N, settings3.sig,
                                        settings3.delta, settings3.A, settings3.m, settings3.Zprimesqrd,
                                        settings3.lambda_B, settings3.kappa_D, settings3.kBT, settings3.xi,
                                        settings3.delta_t, settings3.gaus_var, settings3.random_seed)
        
        # save shit every n
        if i % everyN == 0:
            if write:
                # logging.info(force.temperature(vx,vy,vz,settings3=settings3))
                output.WriteTrajectory3d(fileoutput_eq, i,x,y,z, settings3)
                output.WriteunwrappedState(fileoutput_eq_unwrapped,i,x,y,z,vx,vy,vz)


            hist = update.update_hist(hist, x,y,z,
                                0,settings3.L, 0,settings3.L,0,settings3.L,
                                settings3.dr, settings3.N, settings3.L)
            Ngr += 1 # another position

        g = update.calcg(Ngr,hist, settings3.dr)

                    
    return g