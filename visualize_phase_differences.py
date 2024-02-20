import numpy as np
import matplotlib.pyplot as plt

los = np.load("los.npy")
multi_path =np.load("multipath.npy")

fig, ax = plt.subplots(4, 4)
for i in range(12):
    plt.subplot(4 ,4, i + 1)
    plt.plot(multi_path[:, i])
    plt.plot(los[:, i])
    plt.xlabel("k")
    plt.ylabel("phase difference (rad)")
    plt.legend()

plt.show()