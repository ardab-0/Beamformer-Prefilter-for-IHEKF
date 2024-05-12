import numpy
import matplotlib.pyplot as plt
from matplotlib import pyplot

multipath_amplitude = [0, 0.2, 0.4, 0.6,
                       # 0.8, 1
                       ]
rmse_ip3 = [0.01763632620194625,
           0.02811720002247848,
           0.050747188038465695,
           0.09653643761001325,
           # 0.17269402186006946,
           # 0.2768248859281605
            ]


rmse_ip1 = [0.01763211111940612,
           0.027883982701759943,
           0.050728981685665586,
           0.14340412057480076,
           # 62.32942243142493,
           # 49.58043145000041
            ]

rmse_ip2 = [0.01763211111940612,
           0.02811720002247848,
           0.04933456729968633,
           0.09279774220242953,
           # 8.017401981968579,
           # 0.8144186072135838
            ]

rmse_ihekf = [0.017636326201946256,
              0.023522214274010925,
              0.11666239272753964,
              0.3249921918780431,
              # 0.7894422977188212,
              # 0.9743563971793774
              ]


plt.plot(multipath_amplitude, rmse_ihekf, label="IHEKF")
plt.plot(multipath_amplitude, rmse_ip1, label="IHEKF + IP 1 iteration")
plt.plot(multipath_amplitude, rmse_ip2, label="IHEKF + IP 2 iteration")
plt.plot(multipath_amplitude, rmse_ip3, label="IHEKF + IP 3 iteration")

plt.xlabel("Multipath Amplitude")
plt.ylabel("RMSE (m)")
plt.grid()
plt.legend()
plt.show()