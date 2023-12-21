class Parameters:
    f = 24e9  # Hz
    c = 2.998e8  # m/s
    lmb = c / f  # m
    room_x = [-1.5, 1.5]  # m
    room_y = [-1.5, 1.5]  # m
    room_z = [0, 3]  # m
    sigma_phi = 0.17  # 10  # doesn't affect when measure_phi_m_directly is not set

    sigma = 0.001 # std of complex gaussian noise (input signal noise)
    sigma_a = 1# m/s**2
    dt = 0.1  # s
    N_theta = 200
    N_phi = 200
    N = 1000
    jacobian_type = "numpy"  # "numpy" or "scipy"
    apply_element_pattern = True

    use_multipath = True  # True or False
    multipath_count = 1

    antenna_kind = "square_4_4"  # "original" or "square_4_4" or "irregular_4_4"
    i_list = [ 16]  # antenna number to include at each iteration

    visualize_beampatterns = True
    measure_phi_m_directly = False
    apply_spatial_filter = True

    k = 50 # number of time steps sampled in path

    beamformer_type = "capon" # "fourier" , "capon", "music"

    max_multipath_amplitude = 0.8
    peak_threshold = 0.5 # the amplitude threshold to detect peak in spatial filter
    num_peaks_to_remove = 1 # the number of peaks to remove