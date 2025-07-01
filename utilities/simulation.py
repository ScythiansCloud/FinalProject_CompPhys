import numpy as np
from tqdm import tqdm
import initialize
import force
import update
import settings_task2 as settings
# import settings_task3 as settings
import output


def Simulation(write, Traj_name, everyN):

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
        fileoutput_eq = open(Traj_name + str(everyN) + '_nsteps_' + str(settings.nsteps), "w")
        # fileoutput_prod = open(Traj_name + str(everyN) + '_prod', "w")
        output.WriteTrajectory3d(fileoutput_eq, 0,x,y,z, settings) 
        # output.WriteTrajectory3d(fileoutput_prod, 0,x,y,z) 

    for i in tqdm(range(settings.nsteps)):
        x,y,z, vx, vy, vz = update.update(True, x, y, z, vx, vy, vz, settings.L, settings.N, settings.sig,
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

