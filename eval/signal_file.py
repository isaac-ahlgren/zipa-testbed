import glob
import os
from typing import Callable

import numpy as np
from signal_file_interface import Signal_File_Interface


class Noisy_File(Signal_File_Interface):
    def __init__(self, sf: Signal_File_Interface, target_snr: float):
        self.sf = sf
        self.target_snr = target_snr

        seed = int.from_bytes(os.urandom(8), "big")
        self.rng = np.random.default_rng(seed)

    def calc_snr_dist_params(self, signal: np.ndarray, target_snr: float) -> float:
        """
        Calculate the noise standard deviation for a given signal and target SNR.

        :param signal: The input signal array.
        :param target_snr: The desired signal-to-noise ratio in dB.
        :return: The calculated noise standard deviation.
        """
        sig_sqr_sum = np.mean(signal**2)
        sig_db = 10 * np.log10(sig_sqr_sum)
        noise_db = sig_db - target_snr
        noise_avg_sqr = 10 ** (noise_db / 10)
        return np.sqrt(noise_avg_sqr)

    def add_gauss_noise(self, signal: np.ndarray, target_snr: float) -> np.ndarray:
        """
        Add Gaussian noise to a signal based on a target SNR.

        :param signal: The input signal array.
        :param target_snr: The desired signal-to-noise ratio in dB.
        :return: The signal with added Gaussian noise.
        """

        noise_std = self.calc_snr_dist_params(signal, target_snr)
        noise = self.rng.normal(0, noise_std, len(signal))
        return signal + noise

    def read(self, samples: int) -> np.ndarray:
        buf = self.sf.read(samples)
        return self.add_gauss_noise(buf, self.target_snr)

    def get_finished_reading(self):
        return self.sf.get_finished_reading()

    def get_id(self):
        return self.sf.get_id()

    def reset(self):
        self.sf.reset()

    def sync(self, other_sf):
        self.sf.sync(other_sf.sf)


class Wrap_Around_File(Signal_File_Interface):
    def __init__(
        self,
        sf: Signal_File_Interface,
        wrap_around_limit=None,
    ):
        self.sf = sf
        self.wrap_around_limit = wrap_around_limit
        self.num_of_resets = 0
        self.finished_reading = False

    def read(self, samples: int) -> np.ndarray:
        output = np.array([])

        while not self.finished_reading and samples != 0:
            buf = self.sf.read(samples)
            samples -= len(buf)
            output = np.append(output, buf)

            if self.sf.get_finished_reading():
                if (
                    self.wrap_around_limit is None
                    or self.num_of_resets < self.wrap_around_limit
                ):
                    self.sf.reset()
                else:
                    self.finished_reading = True
                self.num_of_resets += 1
        return output

    def get_finished_reading(self):
        return self.finished_reading

    def get_id(self):
        return self.sf.get_id()

    def reset(self):
        self.num_of_resets = 0
        self.finished_reading = False
        self.sf.reset()

    def sync(self, other_sf):
        self.num_of_resets = other_sf.num_of_resets
        self.sf.sync(other_sf.sf)


class Signal_File(Signal_File_Interface):
    def __init__(
        self,
        signal_directory: str,
        file_names: str,
        load_func: Callable = np.loadtxt,
        id: str = "",
    ):
        """
        Initialize a file-based signal manager that can handle multiple signal files.

        :param signal_directory: The directory containing signal files.
        :param file_pattern: The glob pattern to match files within the directory.
        :param wrap_around_read: If True, wraps around to the first file after the last file is read.
        :param load_func: The function used to load signal data from a file.
        """
        self.signal_directory = signal_directory
        self.files = glob.glob(file_names, root_dir=signal_directory)
        if len(self.files) == 0:
            print("No files found")
        else:
            self.files.sort()
        self.file_index = 0
        self.start_sample = 0
        self.load_func = load_func
        self.curr_file_name = self.signal_directory + self.files[0]
        self.sample_buffer = self.load_func(self.curr_file_name)
        self.dtype = self.sample_buffer.dtype
        self.finished_reading = False
        self.id = id

    def switch_files(self):
        """
        Switch to the next file in the directory or wrap around if enabled.
        """
        self.start_sample = 0
        self.file_index += 1
        if (
            len(self.files) == self.file_index
        ):  # If no more to read, set the finished reading flag
            self.finished_reading = True
        else:
            self.curr_file_name = self.signal_directory + self.files[self.file_index]
            print("Loading in " + self.curr_file_name)
            del self.sample_buffer
            self.sample_buffer = self.load_func(self.curr_file_name)

    def read(self, samples: int) -> np.ndarray:
        """
        Read a specified number of samples across multiple files.

        :param samples: Number of samples to read.
        :return: Array containing the read samples.
        """
        output = np.array([], dtype=self.dtype)

        while samples != 0 and not self.finished_reading:
            samples_can_read = len(self.sample_buffer) - self.start_sample
            if samples_can_read <= samples:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples_can_read
                ]
                output = np.append(output, buffer)
                self.switch_files()
                samples -= samples_can_read
            else:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples
                ]
                output = np.append(output, buffer)
                self.start_sample = self.start_sample + samples
                samples = 0
        return output

    def get_finished_reading(self):
        return self.finished_reading

    def get_id(self):
        return self.id

    def reset(self):
        """
        Reset the reader to the start of the first file.
        """
        if self.curr_file_name != self.signal_directory + self.files[0]:
            if self.sample_buffer is not None:
                del self.sample_buffer
            self.sample_buffer = self.load_func(self.curr_file_name)
        self.finished_reading = False
        self.curr_file_name = self.signal_directory + self.files[0]
        self.start_sample = 0
        self.file_index = 0

    def sync(self, other_sf):
        """
        Synchronize this file reader with another, matching the current file and sample index.

        :param other_sf: Another Signal_File instance to synchronize with.
        """
        self.file_index = other_sf.file_index
        self.start_sample = other_sf.start_sample
        self.curr_file_name = self.signal_directory + self.files[self.file_index]
        del self.sample_buffer
        self.sample_buffer = self.load_func(self.curr_file_name)


class Signal_Buffer(Signal_File_Interface):
    def __init__(self, buf: np.ndarray, id: str = ""):
        """
        Initialize a buffer to manage signal data with optional noise addition.

        :param buf: Array containing the signal data.
        """
        self.signal_buffer = buf
        self.dtype = buf.dtype
        self.start_sample = 0
        self.finished_reading = False
        self.id = id

    def read(self, samples_to_read: int) -> np.ndarray:
        """
        Read a specific number of samples from the buffer, adding noise if specified.

        :param samples_to_read: The number of samples to read from the buffer.
        :return: An array of the read samples, possibly with noise added.
        """

        output = np.array([], dtype=self.dtype)
        while samples_to_read != 0 and not self.get_finished_reading():
            samples_can_read = len(self.signal_buffer) - self.start_sample
            if samples_can_read <= samples_to_read:
                buf = self.signal_buffer[
                    self.start_sample : self.start_sample + samples_can_read
                ]
                output = np.append(output, buf)
                samples_to_read = samples_to_read - samples_can_read
                self.finished_reading = True
            else:
                buf = self.signal_buffer[
                    self.start_sample : self.start_sample + samples_to_read
                ]
                output = np.append(output, buf)
                self.start_sample = self.start_sample + samples_to_read
                samples_to_read = 0
        return output

    def get_finished_reading(self):
        return self.finished_reading

    def get_id(self):
        return self.id

    def sync(self, other_signal_buff: "Signal_Buffer"):
        """
        Synchronize this buffer's index with another signal buffer's index.

        :param other_signal_buff: Another Signal_Buffer instance to synchronize with.
        """
        self.start_sample = other_signal_buff.start_sample

    def reset(self):
        self.finished_reading = False
        self.start_sample = 0
