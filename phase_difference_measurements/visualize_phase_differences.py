import numpy as np
import matplotlib.pyplot as plt
import utils

phi_differences = np.load("phi_differences.npy")
phi_filtered_differences = np.load("phi_filtered_differences.npy")
phi_no_multipath_differences = np.load("phi_no_multipath_differences.npy")

# labels = ["1-2", "1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "1-9", "9-10"]
length = len(phi_differences[0])
r = int(np.ceil(np.sqrt(length)))
fig, ax = plt.subplots(r, r)
for i in range(length):
    plt.subplot(r, r, i + 1)
    plt.plot(phi_differences[:, i], label="no filter")
    plt.plot(phi_no_multipath_differences[:, i], label="no multipath")
    plt.plot(phi_filtered_differences[:, i], label="filtered")
    plt.xlabel("k")
    plt.ylabel("phase difference (rad)")
    plt.legend()





no_filter_error = utils.phase_error(phi_differences, phi_no_multipath_differences)
filter_error = utils.phase_error(phi_filtered_differences, phi_no_multipath_differences)

print(f"No Filter RMSE: {no_filter_error}, Filter RMSE: {filter_error}")
plt.show()
