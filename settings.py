'''In my understanding everything can be written in terms of KbT and sigma, the friciton parameter and the mass (so we set those and everything else is derived...), I hope we get those values'''


## imports
import numpy as np

def init():

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

    global delta 
    delta = 1

    global rho 
    rho = 1

    global Cs 
    Cs = 1 ##maybe we handle this not in settings because we shoudl vary it a lot idk...

    global Z 
    Z = 1

    ## derived constants

    '''L, kappa d, Zprime, A , tau LD'''


    ##########################################################

    #Simulation 
    global N 
    N = 1

    global nsteps
    nsteps = 1

    global nsave
    nsave = 1


    