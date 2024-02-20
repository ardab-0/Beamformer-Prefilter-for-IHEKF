class Parameters:
    f = 24e9  # Hz
    fs = f * 100
    c = 2.998e8  # m/s
    lmb = c / f  # m
    room_x = [-5, 5]  # m
    room_y = [-5, 5]  # m
    room_z = [0, 3]  # m
    sigma_phi = 0.17  #
    sigma = 0.01  # std of complex gaussian noise (input signal noise)
    sigma_a = 1  # m/s**2
    dt = 0.01  # s
    N_theta = 100
    N_phi = 100
    apply_element_pattern = True


    antenna_kind = "square_4_3"  # "original" or "square_4_4" or "irregular_4_4"
    i_list = [5, 8, 12]  # antenna number to include at each iteration

    visualize_beampatterns = True
    measure_phi_m_directly = False
    apply_spatial_filter = False


    beamformer_type = "delay_and_sum"  # "fourier" , "capon", "music"

    peak_threshold = 0.5  # the amplitude threshold to detect peak in spatial filter
    num_peaks_to_remove = 1  # the number of peaks to remove

    target_phi_range_deg = 2
    target_theta_range_deg = 4
    cone_angle = 10
    spatial_filter_initialization_index = 0


VERBOSE = False
