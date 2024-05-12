import numpy as np
import matplotlib.pyplot as plt
import scipy
import utils

foldername = "../test_results/realdata/16/"
# load data

trajectory = np.load(foldername + "0_trajectory.npy")
xs_0 = np.load(foldername + "0_xs.npy")
trajectory_1 = np.load(foldername + "1_trajectory.npy")
xs_1 = np.load(foldername + "1_xs.npy")
trajectory_2 = np.load(foldername + "2_trajectory.npy")
xs_2 = np.load(foldername + "2_xs.npy")
trajectory_3 = np.load(foldername + "3_trajectory.npy")
xs_3 = np.load(foldername + "3_xs.npy")

xs_list = [xs_0, xs_1, xs_2, xs_3]
labels = ["IHEKF", "IHEKF + IP 1 iteration", "IHEKF + IP 2 iteration", "IHEKF + IP 3 iteration"]





ax = plt.axes(projection="3d")
ax.set_xlabel("x(m)")
ax.set_ylabel("y(m)")
ax.set_zlabel("z(m)")
for i, xs in enumerate(xs_list[:2]):
    ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], label=labels[i])

ax.plot3D(trajectory[0, :], trajectory[1, :], trajectory[2, :], "black", linestyle="dotted", label="reference")


plt.legend()
# 2D plot
plt.figure()
for i, s in zip([0, 1, 2], ['X', 'Y', 'Z']):
    ax = plt.subplot(311 + i)
    ax.plot(trajectory.T[:, i], "black", linestyle="dotted", label=s + " - reference")
    ax.grid()
    mean = np.mean(trajectory.T[:, i])
    plt.ylim(mean-1.2, mean+1.2)
    plt.xlabel("steps")
    plt.ylabel(s + "(m)")

for j, xs in enumerate(xs_list[:3]):
    for i, s in zip([0, 1, 2], ['X', 'Y', 'Z']):
        ax = plt.subplot(311 + i)
        ax.plot(xs[:, i], label=s + f" - {labels[j]}")
        plt.xlabel("steps")
        plt.ylabel(s + "(m)")

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
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



plt.show()