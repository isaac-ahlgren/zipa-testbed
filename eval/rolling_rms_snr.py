import numpy as np
from matplotlib import pyplot as plt

device = "mic"
id_num = "231"
hr = "10"

file = f"/mnt/data/{device}_id_10.0.0.{id_num}_date_20240624{hr}.csv"

data = np.genfromtxt(file, delimiter=",")

window_size = 240000
snr_window_size = 240000


def rolling_rms(data, window_size):
    cumsum = np.cumsum(np.insert(data**2, 0, 0))
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    return np.sqrt(window_sums / window_size)


rms_data = rolling_rms(data, window_size)


# Calculate rolling mean and standard deviation for SNR calculation
def rolling_mean_std(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    cumsum_sq = np.cumsum(np.insert(data**2, 0, 0))

    sum_ = cumsum[window_size:] - cumsum[:-window_size]
    sum_sq = cumsum_sq[window_size:] - cumsum_sq[:-window_size]

    mean = sum_ / window_size
    std = np.sqrt(sum_sq / window_size - (mean**2))

    return mean, std


mean, std = rolling_mean_std(rms_data, snr_window_size)

std_squared = np.where(std**2 == 0, np.finfo(float).eps, std**2)
snr_data = 10 * np.log10((mean * 2) / std_squared)

plt.plot(snr_data, label="SNR Data")
# plt.plot(rms_data[:len(snr_data)], label='RMS Data')
plt.legend()
plt.title(f"SNR Data, hr: {hr}")
plt.savefig(f"snr_data/{device}_{id_num}_{hr}.png")
# plt.show()

plt.clf()

plt.title(f"RMS Data, hr: {hr}")
plt.plot(rms_data[: len(snr_data)], label="RMS Data")
plt.legend()
plt.savefig(f"snr_data/{device}_{id_num}_0623{hr}_rms.png")
