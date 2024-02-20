import numpy as np
import matplotlib.pyplot as plt

# los = np.load("los.npy")
multi_path =np.load("multipath.npy")

labels = ["1-2", "1-3", "2-4", "3-5", "4-6", "5-7", "6-8", "1-9", "9-10"]
fig, ax = plt.subplots(4, 4)
for i in range(16):
    plt.subplot(4 , i + 1)
    plt.plot(multi_path[:, i], label="los + multi_path Ant: " + labels[i])
    # plt.plot(los[:, i], label="los Ant: " + labels[i])
    plt.xlabel("k")
    plt.ylabel("phase difference (rad)")
    plt.legend()

plt.show()