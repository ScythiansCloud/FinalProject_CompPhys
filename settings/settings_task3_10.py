'''In my understanding everything can be written in terms of KbT and sigma, the friciton parameter and the mass (so we set those and everything else is derived...), I hope we get those values'''


## imports
import numpy as np

def init(Csi):

    ##########################################################

    #Simulation 
    global N 
    N = 343

    global nsteps_eq
    nsteps_eq = 1000       # nsetps_eq != 500 000 to run the same time when using delta_t=2/1000

    global nsteps   
    nsteps = 1000         # nsteps != 500 000 to run the same time when using delta_t=2/1000 

    global nsave
    nsave = 50

    global kBT
    kBT = 1

    global random_seed
    random_seed = 1213      # != 42069
    ########################################################

    global xi #fricition parameter
    xi = 1

    global m
    m = 1
    
    global sig 
    sig = 1

    global Cs 
    Cs = Csi*sig**(-3) 

    global Z 
    Z = 50

    global gaus_var
    gaus_var = 1

    ## derived constants

    global rho 
    rho = 0.05*sig**(-3)

    global L 
    L = (N/rho)**(1/3)

    global delta 
    delta = sig*1e-2

    global lambda_B 
    lambda_B = sig/100
    
    global kappa_D
    kappa_D = np.sqrt(8*np.pi *lambda_B* Cs)

    global A 
    A = 0.1*kBT

    global Zprime
    Zprime = Z*np.exp(kappa_D*sig/2)/(1+kappa_D*sig/2)
    global Zprimesqrd
    Zprimesqrd = Zprime*Zprime

    global tau
    tau = sig*sig/(kBT)*xi

    global delta_t          # using 2/1000*tau * 500 000 --> corresponds to 100 ...seconds
    delta_t= 2e-3*tau       # changed delta_t = 2/1000*tau which will correspond to the same 
                            # total simulation time as for other concentrations  


    global dr             # bin width for RDF with respect to the box size
    dr = L/2 / 200        # 1000 bins for the RDF




    