import numpy as np
import matplotlib.pyplot as plt

from real_data import DataLoader

#
# def plot_meas_fft(self, decimate=1, plt_zero=True):
#     for idx, raw in zip(self.id_list, self.read_list):
#         raw_fft = np.abs(np.fft.fft(raw, axis=0))
#         stuetz = np.fft.fftfreq(raw.shape[0], 1 / self.ACTUAL_SAMPLING_RATE)
#
#         if plt_zero:
#             raw_fft[0, :] = np.zeros(raw_fft.shape[1])
#
#         max_raw = np.max(raw_fft)
#         min_raw = np.min(raw_fft)
#         fig, axs = plt.subplots(4, 4)
#         fig.suptitle(f"fft on array {idx}")
#
#         for i in range(raw.shape[1]):  # für jede Antennenkombi
#             [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
#             axs[u, v].plot(stuetz[::decimate], raw_fft[::decimate, i], label=f"Channel {i}")
#             axs[u, v].legend()
#             axs[u, v].set_ylim(min_raw, max_raw)
#             axs[u, v].grid("minor")



folder = "./fuer_arda/3/"

tmp = np.load(f"{folder}data.npz")
trajectory_optitrack = tmp["trajectory_optitrack"]
nr_of_meas_points = trajectory_optitrack.shape[0]
REDUCTION_FACTOR = tmp["REDUCTION_FACTOR"]
sample_rate = 3.125e6 / REDUCTION_FACTOR

raw_data = tmp["raw_data"]
raw_data_list = np.array_split(raw_data[:, :, :], nr_of_meas_points, axis=1)
min_length = 100000
for snip in raw_data_list:
    if snip.shape[0] < min_length:
        min_length = snip.shape[1]

cropped_raw_data_list = []
for snip in raw_data_list:
    cropped_raw_data_list.append(snip[:, :min_length])
raw_data_list = cropped_raw_data_list
del cropped_raw_data_list
raw_data = np.stack(raw_data_list, axis=0)

print(raw_data.shape)
TIME_STEPS, ARRAY_COUNT, SAMPLE_NR, CHANNEL_NR = raw_data.shape

time_step = 0
array_nr = 2

data = raw_data[time_step, array_nr]


fig, axs = plt.subplots(4, 4)
fig.suptitle(f"Raw data for time step {time_step}, array: {array_nr}")
for i in range(data.shape[1]):
    [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
    axs[u, v].plot(data[:, i], label=f"Channel {i}")
    axs[u, v].legend()
    axs[u, v].grid("minor")


raw_fft = np.abs(np.fft.fft(data, axis=0))

fig, axs = plt.subplots(4, 4)
fig.suptitle(f"FFT of data for time step {time_step}, array: {array_nr}")
for i in range(data.shape[1]):
    [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
    axs[u, v].plot(raw_fft[:, i], label=f"Channel {i}")
    axs[u, v].legend()
    axs[u, v].grid("minor")

amplitude = np.linalg.norm(data, axis=0)

fig, axs = plt.subplots(4, 4)
fig.suptitle(f"Amplitude of data for time step {time_step}, array: {array_nr}")
for i in range(data.shape[1]):
    [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
    axs[u, v].plot(amplitude[ i], label=f"Channel {i}")
    axs[u, v].legend()
    axs[u, v].grid("minor")


fft_peaks = np.zeros((TIME_STEPS, CHANNEL_NR))
for t_idx in range(raw_data.shape[0]):
    for channel_nr in range(CHANNEL_NR):
        data = raw_data[t_idx, array_nr]
        raw_fft = np.abs(np.fft.fft(data, axis=0))
        max_val = np.max(raw_fft[:, channel_nr])
        fft_peaks[t_idx, channel_nr] = max_val

plt.figure()
plt.title(f"FFT peaks for antenna arraz: {array_nr}")
max = np.max(fft_peaks, axis=1)
fft_peaks /= max.reshape((-1, 1))
plt.imshow(fft_peaks)
plt.xlabel("Channels")
plt.ylabel("Time step")


amplitude_peaks = np.zeros((TIME_STEPS, CHANNEL_NR))
for t_idx in range(raw_data.shape[0]):
    data = raw_data[t_idx, array_nr]
    amplitude = np.linalg.norm(data, axis=0)
    for channel_nr in range(CHANNEL_NR):
        amplitude_peaks[t_idx, channel_nr] = amplitude[channel_nr]

plt.figure()
plt.title(f"Amplitude for antenna array: {array_nr}")
max = np.max(amplitude_peaks, axis=1)
amplitude_peaks /= max.reshape((-1, 1))
plt.imshow(amplitude_peaks)

# normalization_coefficient = np.mean(fft_peaks, axis=0)
# np.save(f"normalization_coef_array{array_nr}.npy", normalization_coefficient)
#
#
#
# # normalization
# data = raw_data[time_step, array_nr]
# raw_fft = np.abs(np.fft.fft(data, axis=0))
# fig, axs = plt.subplots(4, 4)
# fig.suptitle(f"Normalized FFT of data for time step {time_step}, array: {array_nr}")
# for i in range(data.shape[1]):
#     [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
#     axs[u, v].plot(raw_fft[:, i] / normalization_coefficient[i], label=f"Channel {i}")
#     axs[u, v].legend()
#     axs[u, v].grid("minor")


plt.show()
