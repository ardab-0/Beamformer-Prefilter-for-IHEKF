from dataclasses import dataclass


@dataclass
class Parameters:
    f: float = 24e9  # Hz
    fs: float = f * 100
    c: float = 2.998e8  # m/s
    lmb: float = c / f  # m
    room_x: tuple = (-5, 5)  # m
    room_y: tuple = (-5, 5)  # m
    room_z: tuple = (-5, 5)  # m
    sigma_phi: float = 0.17  # 10  # doesn't affect when measure_phi_m_directly is not set
    sigma: float = 0.01  # std of complex gaussian noise (input signal noise)
    sigma_a: float = 8  # m/s**2
    dt: float = 0.01  # s
    N_theta: int = 200
    N_phi: int = 200
    N: int = 100 # number of samples time vector
    jacobian_type: str = "numpy"  # "numpy" or "scipy"
    apply_element_pattern: bool = True

    use_multipath: bool = True  # True or False
    multipath_count: int = 4

    antenna_kind: str = "irregular_8_2"  # "original" or "square_4_4" or "irregular_4_4"
    i_list: tuple = (12, 14, 16)  # antenna number to include at each iteration

    visualize_beampatterns: bool = False
    measure_phi_m_directly: bool = False
    apply_spatial_filter: bool = True

    k: int = 50  # number of time steps sampled in path

    beamformer_type: str = "delay_and_sum"  # "fourier" , "capon", "music"

    min_multipath_amplitude: float = 0.2
    max_multipath_amplitude: float = 0.8
    peak_threshold: float = 0.5  # the amplitude threshold to detect peak in spatial filter
    num_peaks_to_remove: int = 1  # the number of peaks to remove

    target_phi_range_deg: float = 2
    target_theta_range_deg: float = 4
    cone_angle: float = 10
    spatial_filter_initialization_index: int = 0




