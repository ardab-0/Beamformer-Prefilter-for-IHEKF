import numpy as np
from settings.config import Parameters as params
import utils


def measure_multipath(antenna_positions, beacon_pos, sigma_phi, multipath_count):
    phi_mix = np.random.rand() * 2 * np.pi
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = -2 * np.pi * params.f * tau + phi_mix
    s_mix = np.exp(1j * phi)
    multipath_sources = []
    for p in range(multipath_count):
        x = np.random.uniform(params.room_x[0], params.room_x[1])
        y = np.random.uniform(params.room_y[0], params.room_y[1])
        z = np.random.uniform(params.room_z[0], params.room_z[1])
        a = np.random.uniform(0, 1)
        multipath_sources.append({"position": [x, y, z],
                                  "amplitude": a})
        multipath_source_pos = np.array([[x, y, z]]).T
        tau = np.linalg.norm(antenna_positions - multipath_source_pos, axis=0) / params.c
        phi = -2 * np.pi * params.f * tau + phi_mix
        s_mix += a * np.exp(1j * phi)

    total_phi = np.angle(s_mix)
    n = np.random.randn(*total_phi.shape) * sigma_phi
    total_phi = utils.mod_2pi(total_phi + n).reshape((-1, 1))
    return total_phi, multipath_sources


def measure(antenna_positions, beacon_pos, sigma_phi):
    # random phase offset for receiver
    phi_mix = np.random.rand() * 2 * np.pi
    # phi_mix = 0
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c

    n = np.random.randn(*tau.shape) * sigma_phi
    phi = utils.mod_2pi(-2 * np.pi * params.f * tau + phi_mix + n).reshape((-1, 1))
    return phi


def measure_s_m(t, antenna_positions, beacon_pos, phi_B, sigma):
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
    x = np.exp(1j * phi)
    n = np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    return x + sigma * n  # phase of s_m is gaussian due to signal being non-zero


def measure_s_m_multipath(t, antenna_positions, beacon_pos, phi_B, sigma, multipath_sources):
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
    s_m = np.exp(1j * phi)

    for p, source in enumerate(multipath_sources):
        x = source["x"]
        y = source["y"]
        z = source["z"]
        a = source["a"]
        multipath_source_pos = np.array([[x, y, z]]).T
        tau = np.linalg.norm(antenna_positions - multipath_source_pos, axis=0) / params.c
        phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
        s_m += a * np.exp(1j * phi)
        print(f"Multipath {p}: a:{a}, x:{x}, y:{y}, z:{z}")

    n = np.random.randn(*s_m.shape) + 1j * np.random.randn(*s_m.shape)
    return s_m + sigma * n


def generate_multipath_sources(multipath_count):
    sources = []
    for p in range(multipath_count):
        x = np.random.uniform(params.room_x[0]/2, params.room_x[1]/2)
        y = np.random.uniform(params.room_y[0]/2, params.room_y[1]/2)
        z = np.random.uniform(-params.room_z[1], params.room_z[1])
        a = np.random.uniform(0, params.max_multipath_amplitude)
        sources.append({"a": a,
                        "x": x,
                        "y": y,
                        "z": z})

    return sources


def measure_phi(s_m, f_m, t):  # might need to modify mean
    phi_lo = np.random.rand() * 2 * np.pi
    s_lo = np.exp(-1j * (2 * np.pi * f_m * t + phi_lo))
    s_mix = s_m * s_lo
    phi = np.mean(np.angle(s_mix), axis=1).reshape((-1, 1))
    phi = utils.mod_2pi(phi)
    return phi


def compute_G(dt):
    G = np.array(
        [
            [dt ** 2 / 2, 0, 0],
            [0, dt ** 2 / 2, 0],
            [0, 0, dt ** 2 / 2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ]
    )

    return G


def compute_F(dt):
    F = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, dt],
        ]
    )

    return F


def get_A_full(antenna_pos):
    i_max = len(antenna_pos[0]) - 1
    A_full = np.zeros((i_max, len(antenna_pos[0])))

    for i in range(i_max):
        A_full[i, i] = 1
        A_full[i, i + 1] = -1

    return A_full


def compute_phase_shift(x, f, u, r):
    return x * np.exp(1j * 2 * np.pi * f * np.dot(u, r) / params.c)
