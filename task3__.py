from utilities.simulation import Simulation3
import numpy as np
from utilities import utils
import settings.settings_task3_10 as settings10
import settings.settings_task3 as settings
import matplotlib.pyplot as plt

# Set up output folder creation and logging
output_dir = utils.create_output_directory()
utils.setup_logging(output_dir)
print('sdf')
gr100 = Simulation3(output_dir, True, 'testCS_333', 10, 3567655678, settings,333)

plt.plot(gr100)
plt.show()
print('now we save g>(r)')
g_r_save = open(output_dir / 'gofr', "w")

for i in range(len(gr100)):
    g_r_save.write(str(gr100[i])+ '\n')
# g_r_save.write('\n')