'''this will be a cleaned programm wich will do the entire task 2 in 1 go'''

#imports
import numpy as np
from utilities.simulation import Simulation2
import settings.settings_task2 as settings
import matplotlib.pyplot as plt
from utilities import utils

#create output folder 
output_dir = utils.create_output_directory()
utils.setup_logging(output_dir)

# do run
settings.init(1) # salt concentration is arbitrary as potentials are turned off
Simulation2(output_dir, True, "Try02", everyN=10)

# read out the unwrapped coordinates and calculate kinetic energy and MSD over time



#plot Enery and MSD over time in the output dir ##