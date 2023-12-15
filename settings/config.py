class Parameters:
    f = 24e9  # Hz
    c = 2.998e8  # m/s
    lmb = c / f  # m
    room_x = [-1.5, 1.5]  # m
    room_y = [-1.5, 1.5]  # m
    room_z = [0, 3]  # m
    sigma_phi = 0.17  # 10  # doesn't affect when measure_phi_m_directly is not set

    sigma = 0.01 # std of complex gaussian noise (input signal noise)
    sigma_a = 0.4 # m/s**2
    dt = 0.1  # s
    N_theta = 50
    N_phi = 100
    N = 500
    jacobian_type = "numpy"  # "numpy" or "scipy"
    apply_element_pattern = True

    use_multipath = True  # True or False
    multipath_count = 1

    antenna_kind = "square_4_4"  # "original" or "square_4_4" or "irregular_4_4"
    i_list = [10]  # antenna number to include at each iteration

    visualize_beampatterns = True
    measure_phi_m_directly = False
    apply_spatial_filter = False

    k = 600 # number of time steps sampled in path
