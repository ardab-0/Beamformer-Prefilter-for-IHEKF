class Parameters:
    f = 24e9  # Hz
    c = 2.998e8  # m/s
    lmb = c / f  # m
    room_x = [-1.5, 1.5]  # m
    room_y = [-1.5, 1.5]  # m
    room_z = [0, 3]  # m
    multipath_count = 0
    i_list = [16]  # antenna number to include at each iteration
    sigma_phi = 0.17  # 10 deg
    sigma_a = 1  # m/s**2
    dt = 0.1  # s
    N_theta = 1000
    N = 1500