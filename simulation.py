import numpy as np
import initialize
import force
import update
import settings_task2 as settings
# import settings_task3 as settings
import output


def Simulation(write, Traj_name, everyN):
    # print(f"After:\n rho = {settings.rho:.2e}")
    # print(f"L = {settings.l}")

    # random seed for reproducibility
    x, y, z, vx, vy, vz = initialize.InitializeAtoms()
    fx, fy, fz = force.acc(x, y, z, settings.l, settings.N, settings.sig, settings.delta, settings.A, settings.m,
                                      settings.Zprimesqrd, settings.lambda_B, settings.kappa_D, settings.kbT)
    

    # rescale velocities should not be needed here, as we do this already in initialize 
    # T_curr = initialize.temperature(vx, vy, vz)
    # vx, vy, vz = initialize.rescalevelocity(vx, vy, vz, settings.kBT, T_curr)
    
    print(f'vx, vy, vz = {vx[0], vy[0], vz[0]}')
    print(f'fx, fy, fz = {fx[0], fy[0], fz[0]}')


        # open documents for eq run
    if write:
        fileoutputeq = open(Traj_name + str(everyN) + '_eq', "w")
        # fileoutput = open(Traj_name + str(everyN) + '_prod', "w")
        output.WriteTrajectory3d(fileoutputeq, 0,x,y,z) # noch anpassen an L, Lz
