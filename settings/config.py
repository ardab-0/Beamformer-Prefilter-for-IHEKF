class Parameters:
    f = 24e9  # Hz
    fs = f * 100
    c = 2.998e8  # m/s
    lmb = c / f  # m
    room_x = [-5, 5]  # m
    room_y = [-5, 5]  # m
    room_z = [0, 3]  # m
    sigma_phi = 0.17  # 10

    sigma = 0.01  # std of complex gaussian noise (input signal noise)
    sigma_a = 5  # m/s**2
    dt = 0.01  # s
    N_theta = 100
    N_phi = 100
    N = 200
    jacobian_type = "numpy"  # "numpy" or "scipy"
    apply_element_pattern = False

    use_multipath = True  # True or False
    multipath_count = 3
    multipath_amplitude = 0.2

    antenna_kind = "square_8_8"  # "original" or "square_4_4" or "irregular_4_4"
    i_list = [64]  # antenna number to include at each iteration

    visualize_beampatterns = True
    measure_phi_m_directly = False
    apply_spatial_filter = False

    k = 200  # number of time steps sampled in path

    beamformer_type = "delay_and_sum"  # "fourier" , "capon", "music"

    min_multipath_amplitude = 0.2
    max_multipath_amplitude = 0.8
    peak_threshold = 0.3  # the amplitude threshold to detect peak in spatial filter
    num_peaks_to_remove = 1  # the number of peaks to remove

    target_phi_range_deg = 2
    target_theta_range_deg = 4
    cone_angle = 10
    spatial_filter_initialization_index = 0
    max_iteration = 3

VERBOSE = True
