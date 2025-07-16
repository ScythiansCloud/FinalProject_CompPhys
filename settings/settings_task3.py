'''In my understanding everything can be written in terms of KbT and sigma, the friciton parameter and the mass (so we set those and everything else is derived...), I hope we get those values'''


## imports
import numpy as np

def init(Csi):

    ##########################################################

    #Simulation 
    global N 
    N = 343

    global nsteps_eq
    nsteps_eq = 100000       # nsetps_eq != 100 000 for equilibration run

    global nsteps   
    nsteps = 100000          # nsteps != 100 000 for production run

    global nsave
    nsave = 10

    global kBT
    kBT = 1

    global random_seed
    random_seed = 1345678     # != 42069
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

    global delta_t          # using 1/100*tau * 100 000 --> corresponds to 100 ...seconds
    delta_t= 1e-2*tau       # will change for Cs10

    global dr             # bin width for RDF with respect to the box size
    dr = L/2 / 200#500        # 500 bins for the RDF




    