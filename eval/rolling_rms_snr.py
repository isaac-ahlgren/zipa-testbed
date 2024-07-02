import numpy as np
from matplotlib import pyplot as plt 
import statistics as st

file = 'mic_id_10.0.0.227_date_2024062006.csv'

data = np.array([])
data = np.genfromtxt(file, delimiter=',')

window_size = 20000
snr_window_size = 20000

rms_data = np.zeros(len(data) - window_size + 1)
snr_data = np.zeros(len(rms_data) - snr_window_size + 1)

# rolling RMS code 
for i in range(len(data) - window_size + 1):
    window = data[i:i + window_size]
    rms = np.sqrt(np.mean(window**2))

    rms_data[i] = rms


# rolling snr code in respect to whatever you make it respect ig 
for j in range(len(rms_data) - snr_window_size + 1):
    snr_window = rms_data[j:j + snr_window_size]
    mean = st.mean(snr_window)
    std = st.stdev(snr_window)
    std_squared = np.where(std**2 == 0, np.finfo(float).eps, std**2)
    snr = 10 * np.log10((mean * 2) / std_squared)

    snr_data[j] = snr


plt.plot(snr_data)
plt.plot(rms_data)
plt.savefig('snr_data/naming_convention.png')
plt.show()