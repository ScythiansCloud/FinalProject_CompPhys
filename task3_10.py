from utilities.simulation import Simulation3
import numpy as np
from utilities import utils
import settings.settings_task3_10 as settings10
import matplotlib.pyplot as plt

# Set up output folder creation and logging
output_dir = utils.create_output_directory()
utils.setup_logging(output_dir)

gr10 = Simulation3(output_dir, True, 'testCS_10', 100, 111222333, settings10, 10)

plt.plot(gr10)
plt.show()

g_r_save = open(output_dir / 'gr', "w")

for i in range(len(gr10)):
    g_r_save.write(str(gr10[i])+ ' ')