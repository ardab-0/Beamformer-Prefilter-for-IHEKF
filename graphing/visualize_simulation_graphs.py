import numpy as np
import matplotlib.pyplot as plt
import scipy
import utils

foldername = "../test_results/simulation_data2/"
# load data

# same trajectory for all cases
trajectory = np.load(foldername + "ip1_0.0_trajectory.npy")
xs_list = []
for i in range(4):
    xs = np.load(foldername + f"ip1_{(2 * i)/10}_xs.npy")
    xs_list.append(xs)

labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

ax = plt.axes(projection="3d")
ax.set_xlabel("x(m)")
ax.set_ylabel("y(m)")
ax.set_zlabel("z(m)")
for i, xs in enumerate(xs_list):
    ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], label=labels[i])

ax.plot3D(trajectory[0, :], trajectory[1, :], trajectory[2, :], "black", linestyle="dotted", label="reference")

plt.legend()
# 2D plot
plt.figure()

for i, s in zip([0, 1, 2], ['X', 'Y', 'Z']):
    ax = plt.subplot(311 + i)
    ax.plot(trajectory.T[:, i], "black", linestyle="dotted", label=s + " - reference")
    ax.legend()
    ax.grid()
    plt.xlabel("steps")
    plt.ylabel(s + "(m)")

for j, xs in enumerate(xs_list[-2:]):
    for i, s in zip([0, 1, 2], ['X', 'Y', 'Z']):
        ax = plt.subplot(311 + i)
        ax.plot(xs[:, i], label=s + f" - {labels[j]}")
        ax.legend()
        ax.grid()
        plt.xlabel("steps")
        plt.ylabel(s + "(m)")

# compute ecdf
plt.figure()
plt.grid()
for xs in xs_list:
    error_vect = utils.error_vector(xs[:, :3].T, trajectory)
    res = scipy.stats.ecdf(error_vect)
    ax = plt.subplot()
    res.cdf.plot(ax)
    print("3D RMSE: ", utils.rmse(xs[:, :3].T, trajectory))
    print("2D RMSE: ", utils.rmse(xs[:, :2].T, trajectory[:2]))


plt.legend(labels)
ax.set_xlabel('Distance Error (m)')
ax.set_ylabel('Cumulative Error Function')

plt.figure()
for xs in xs_list:
    error_vect = utils.error_vector(xs[:, :3].T, trajectory)
    plt.plot(error_vect)

plt.legend(labels)
plt.xlabel('Time Step')
plt.ylabel('Deviation (m)')
plt.show()
