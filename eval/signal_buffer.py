import numpy as np
from eval_tools import calc_snr_dist_params

class Signal_Buffer:
    def __init__(self, buf: np.ndarray, noise: bool = False, target_snr: int = 20):
        """
        Initialize a buffer to manage signal data with optional noise addition.

        :param buf: Array containing the signal data.
        :param noise: If true, Gaussian noise is added to the output based on the target SNR.
        :param target_snr: The signal-to-noise ratio used to calculate noise level.
        """
        self.signal_buffer = buf
        self.index = 0
        self.noise = noise
        if self.noise:
            self.noise_std = calc_snr_dist_params(buf, target_snr)

    def read(self, samples_to_read: int) -> np.ndarray:
        """
        Read a specific number of samples from the buffer, adding noise if specified.

        :param samples_to_read: The number of samples to read from the buffer.
        :return: An array of the read samples, possibly with noise added.
        """
        samples = samples_to_read

        output = np.array([])
        while samples_to_read != 0:
            samples_can_read = len(self.signal_buffer) - self.index
            if samples_can_read <= samples_to_read:
                buf = self.signal_buffer[self.index : self.index + samples_can_read]
                output = np.append(output, buf)
                samples_to_read = samples_to_read - samples_can_read
                self.index = 0
            else:
                buf = self.signal_buffer[self.index : self.index + samples_to_read]
                output = np.append(output, buf)
                self.index = self.index + samples_to_read
                samples_to_read = 0
        if self.noise:
            noise = np.random.normal(0, self.noise_std, samples)
            output += noise
        return output

    def sync(self, other_signal_buff: "Signal_Buffer"):
        """
        Synchronize this buffer's index with another signal buffer's index.

        :param other_signal_buff: Another Signal_Buffer instance to synchronize with.
        """
        self.index = other_signal_buff.index

    def reset(self):
        self.index = 0