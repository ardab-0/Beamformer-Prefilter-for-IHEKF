import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



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



folder = "./fuer_arda/1/"
coefficient_save_folder = "./amplitude_normalization_coefficients/"
time_step = 0
array_nr = 0



tmp = np.load(f"{folder}Calibration_data.npz")
raw_data = tmp["raw_data"]



print(raw_data.shape)
TIME_STEPS, ARRAY_COUNT, SAMPLE_NR, CHANNEL_NR = raw_data.shape
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


fft_peaks = np.zeros((TIME_STEPS, CHANNEL_NR))
for t_idx in range(raw_data.shape[0]):
    for channel_nr in range(CHANNEL_NR):
        data = raw_data[t_idx, array_nr]
        raw_fft = np.abs(np.fft.fft(data, axis=0))
        max_val = np.max(raw_fft[:, channel_nr])
        fft_peaks[t_idx, channel_nr] = max_val

plt.figure()
plt.title(f"FFT peaks for antenna arraz: {array_nr}")
fft_peaks /= fft_peaks[:, 13].reshape((-1, 1)) # use channel 0 for normalization
plt.imshow(fft_peaks)
plt.xlabel("Channels")
plt.ylabel("Time step")


normalization_coefficient = np.mean(fft_peaks, axis=0)
# np.save(f"{coefficient_save_folder}normalization_coef_array{array_nr}.npy", normalization_coefficient)



# normalization
data = raw_data[time_step, array_nr]
raw_fft = np.abs(np.fft.fft(data, axis=0))
fig, axs = plt.subplots(4, 4)
fig.suptitle(f"Normalized FFT of data for time step {time_step}, array: {array_nr}")
for i in range(data.shape[1]):
    [v, u] = [np.mod(i, 4), int(np.floor(i / 4))]
    axs[u, v].plot(raw_fft[:, i] / normalization_coefficient[i], label=f"Channel {i}")
    axs[u, v].legend()
    axs[u, v].grid("minor")


plt.show()
