'''In my understanding everything can be written in terms of KbT and sigma, the friciton parameter and the mass (so we set those and everything else is derived...), I hope we get those values'''


## imports
import numpy as np

def init(Csi):

    ##########################################################

    #Simulation 
    global N 
    N = 1

    global nsteps
    nsteps = 1

    global nsave
    nsave = 1

    # constants
    global kB
    kB = 1
    ########################################################

    # system constants
    global T
    T = 1

    global xi #fricition parameter
    xi = 1

    global m
    m = 1
    
    global sig 
    sig = 1

    global Cs 
    Cs = Csi ##maybe we handle this not in settings because we shoudl vary it a lot idk...

    global Z 
    Z = 1

    global gaus_var
    gaus_var = 1

    ## derived constants

    global rho 
    rho = 0.005*sig**(-3)

    global L 
    L = (N/rho)**(1/3)

    global delta 
    delta = sig*1e-2

    global lambda_B 
    lambda_B = sig/100
    
    global kappa_D
    kappa_D = np.sqrt(8*np.pi *lambda_B* Csi  )

    global A 
    A = 0.1*kB*T

    global Zprime
    Zprime = Z*np.exp(kappa_D*sig/2)/(1+kappa_D*sig/2)

    global tau
    tau = sig*sig/(kB*T)*xi

    global delta_t
    delta_t= 1e-3*tau

    '''L, kappa d, Zprime, A , tau LD, lambda_B'''





    