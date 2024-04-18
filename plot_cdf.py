import matplotlib.pyplot as plt
import numpy as np
import scipy

save_directory = "test_results/realdata_error_vector"
# files = ["04-17-2024_10-44-15.npy",
#          "04-17-2024_10-54-25.npy",
#          "04-17-2024_11-02-34.npy",
#          "04-17-2024_11-11-06.npy"]

# files = ["04-17-2024_11-12-46.npy",
#          "04-17-2024_11-20-26.npy",
#          "04-17-2024_11-39-39.npy",
#          "04-17-2024_11-48-45.npy"]

files = ["04-17-2024_12-03-53.npy",
         "04-17-2024_12-10-01.npy",
         "04-17-2024_12-26-09.npy",
         "04-17-2024_12-42-19.npy"]

labels = ["IHEKF", "IHEKF + IP 1it", "IHEKF + IP 2it", "IHEKF + IP 3it"]

ax = plt.subplot()

for filename in files:
    error_vect = np.load(save_directory + "/" + filename)
    res = scipy.stats.ecdf(error_vect)
    res.cdf.plot(ax)

plt.legend(labels)
ax.set_xlabel('Distance Error (m)')
ax.set_ylabel('Cumulative Error Function')

plt.show()