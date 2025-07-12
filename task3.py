import numpy as np
import matplotlib.pyplot as plt
from utilities.simulation import Simulation3
import settings.settings_task3 as settings3
import logging
from utilities import utils
import output


def main():
    output_dir = utils.create_output_directory()
    utils.setup_logging(output_dir)
    logging.info('Task 3 started')
    Cs_list = [10, 100, 333, 666, 1000] # Salt conentraction which will be multiplied by sigma=1

    g_of_r_list = []

    for i in range(len(Cs_list)):
        logging.info(f"Starting simulation with salt conentration of Cs={Cs_list[i]}.")
        settings3.init(Cs_list[i])
        g_of_r = Simulation3(output_dir, True, f"Task3_Cs{Cs_list[i]}", 10, settings3.random_seed, settings3)
        g_of_r_list.append(g_of_r)



    #### Plotting Stuff #####
    bin_dr_arr = np.linspace(0, settings3.L/2, len(g_of_r)) + 0.5*settings3.dr

    plt.figure(figsize=(8,5))

    plt.plot(bin_dr_arr, g_of_r_list[0], label=r"$g(r)$ 10")  

    plt.plot(bin_dr_arr, g_of_r_list[1], label=r"$g(r)$ 100")  
    plt.plot(bin_dr_arr, g_of_r_list[2], label=r"$g(r)$ 333")  
    plt.plot(bin_dr_arr, g_of_r_list[3], label=r"$g(r)$ 666")  
    plt.plot(bin_dr_arr, g_of_r_list[4], label=r"$g(r)$ 1000")  
    plt.xlabel("r")
    plt.ylabel("g(r)")
    # plt.xlim(1.5, settings3.L/2)
    plt.ylim(0, 6)
    plt.legend()
    # plt.title("")
    plt.tight_layout()
    # plt.savefig('part_c_energies.pdf', dpi=150)