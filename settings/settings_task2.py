'''In my understanding everything can be written in terms of KbT and sigma, the friciton parameter and the mass (so we set those and everything else is derived...), I hope we get those values'''


## imports
import numpy as np

def init(Csi):

    ##########################################################

    #Simulation 
    global N 
    N = 256

    global nsteps
    nsteps = 200000

    global nsave
    nsave = 10

    # constants
    global kBT
    kBT = 1

    global random_seed
    random_seed = 42069161
    ########################################################

    global xi #fricition parameter
    xi = 1

    global m
    m = 1
    
    global sig 
    sig = 1

    global Cs 
    Cs = Csi 

    global Z 
    Z = 50

    global gaus_var
    gaus_var = 1

    ## derived constants

    global rho 
    rho = 0.5*sig**(-3)

    global L 
    L = (N/rho)**(1/3)

    global delta 
    delta = sig*1e-2

    global lambda_B 
    lambda_B = sig/100
    
    global kappa_D
    kappa_D = np.sqrt(8*np.pi *lambda_B* Csi  )

    global A 
    A = 0.1*kBT

    global Zprime
    Zprime = Z*np.exp(kappa_D*sig)/(1+kappa_D*sig/2)
    global Zprimesqrd
    Zprimesqrd = Zprime*Zprime

    global tau
    tau = sig*sig/(kBT)*xi

    global delta_t
    delta_t= 1e-3*tau

    global tau_berendsen
    tau_berendsen = 1000*delta_t



'''Some of the parameters are only needed for task 3 but they are included here anyway :)'''






    