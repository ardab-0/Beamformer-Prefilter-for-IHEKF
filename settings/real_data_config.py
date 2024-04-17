class Parameters:
    f = 24e9  # Hz
    fs = f * 100  # not used in real data
    c = 2.998e8  # m/s
    lmb = c / f  # m
    sigma_phi = 0.52  # phase noise in radian  (30 deg)
    sigma_a = 9.81 / 10  # m/s**2
    sigma_x0 = 0.01  # initial position uncertainty
    sigma_v0 = 0.01  # initial velocity uncertainty
    N = 7000
    dt = 0.01  # s
    N_theta = 100
    N_phi = 100
    apply_element_pattern = False  # applz element pattern in visulization

    antenna_kind = "2_6-3-8-d=1"  # "original" or "square_4_4" or "irregular_4_4"
    i_list = [12, 14, 16]  # antenna number to include at each iteration

    visualize_beampatterns = False
    apply_spatial_filter = False

    beamformer_type = "delay_and_sum"  # "fourier" , "capon", "music"
    peak_threshold = 0.5  # the amplitude threshold to detect peak in spatial filter
    num_peaks_to_remove = 1  # the number of peaks to remove

    cone_angle = 5

    iteration_count = 1 # number of iterations in iterative filter
    data_length_ratio = 1
    step = 60  # subsampling step for raw data at each iteration
    folder = "./fuer_arda/5"


VERBOSE = True
