import glob
import os
from typing import Callable

import numpy as np
from signal_file_interface import Signal_File_Interface


class Noisy_File(Signal_File_Interface):
    def __init__(self, sf: Signal_File_Interface, target_snr: float, seed=None):
        self.sf = sf
        self.target_snr = target_snr

        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(self.seed)

    def calc_snr_dist_params(self, signal: np.ndarray, target_snr: float) -> float:
        """
        Calculate the noise standard deviation for a given signal and target SNR.

        :param signal: The input signal array.
        :param target_snr: The desired signal-to-noise ratio in dB.
        :return: The calculated noise standard deviation.
        """
        sig_sqr_sum = np.mean(signal**2)
        if sig_sqr_sum != 0:
            sig_db = 10 * np.log10(sig_sqr_sum)
            noise_db = sig_db - target_snr
            noise_avg_sqr = 10 ** (noise_db / 10)
            output = np.sqrt(noise_avg_sqr)
        else:
            output = 0
        return output

    def add_gauss_noise(self, signal: np.ndarray, target_snr: float) -> np.ndarray:
        """
        Add Gaussian noise to a signal based on a target SNR.

        :param signal: The input signal array.
        :param target_snr: The desired signal-to-noise ratio in dB.
        :return: The signal with added Gaussian noise.
        """

        if self.seed is None:
            self.seed = int.from_bytes(os.urandom(8), "big")
            self.rng = np.random.default_rng(self.seed)

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

    def get_global_index(self):
        return self.sf.global_index

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def get_seed(self):
        return self.seed

    def regen_seed(self):
        self.seed = int.from_bytes(os.urandom(8), "big")
        self.rng = np.random.default_rng(self.seed)

    def set_global_index(self, index):
        self.sf.set_global_index(index)

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
        self.global_index = 0
        self.finished_reading = False

    def read(self, samples: int) -> np.ndarray:
        output = np.array([], dtype=self.sf.dtype)

        while not self.finished_reading and samples != 0:
            buf = self.sf.read(samples)
            self.global_index += len(buf)
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

    def get_global_index(self):
        return self.global_index

    def set_global_index(self, index):
        self.num_of_resets = 0
        self.sf.set_global_index(index)
        if self.sf.get_finished_reading():
            subfile_index = index % self.sf.max_length
            self.sf.set_global_index(subfile_index)
            self.num_of_resets = index // self.sf.max_length
            if self.num_of_resets >= self.wrap_around_limit:
                self.finished_reading = True
            else:
                self.finished_reading = False
        else:
            self.finished_reading = False
        self.global_index = index

    def reset(self):
        self.num_of_resets = 0
        self.global_index = 0
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
        print(f"{signal_directory}/{file_names}")
        if len(self.files) == 0:
            print("No files found")
        else:
            self.files.sort()
        self.file_index = 0
        self.start_sample = 0
        self.global_index = np.uint64(0)
        self.file_global_indexes = [0]
        self.load_func = load_func
        self.curr_file_name = self.signal_directory + self.files[0]
        self.sample_buffer = None
        self.dtype = None
        self.finished_reading = False
        self.max_length = None
        self.id = id

        self.gen_file_lookup_table()

    def load_in_buf(self, file_index):
        self.file_index = file_index
        self.curr_file_name = self.signal_directory + self.files[file_index]
        print("Loading in " + self.curr_file_name)
        del self.sample_buffer
        self.sample_buffer = self.load_func(self.curr_file_name)
        self.dtype = self.sample_buffer.dtype

    def switch_files(self):
        """
        Switch to the next file in the directory or wrap around if enabled.
        """
        file_index = self.file_index + 1
        if (
            len(self.files) == file_index
        ):  # If no more to read, set the finished reading flag
            self.finished_reading = True
        else:
            self.start_sample = 0
            self.load_in_buf(file_index)
            self.add_file_global_index(self.global_index)

    def read(self, samples: int) -> np.ndarray:
        """
        Read a specified number of samples across multiple files.

        :param samples: Number of samples to read.
        :return: Array containing the read samples.
        """
        write_pos = 0

        if self.sample_buffer is None:  # Loading files into ram only when necessary
            self.sample_buffer = self.load_func(self.curr_file_name)
            self.dtype = self.sample_buffer.dtype

        output = np.zeros(samples, dtype=self.dtype)

        while samples != 0 and not self.finished_reading:
            samples_can_read = len(self.sample_buffer) - self.start_sample

            if samples_can_read <= samples:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples_can_read
                ]
                output[write_pos : write_pos + samples_can_read] = buffer
                self.global_index += samples_can_read
                self.switch_files()
                samples -= samples_can_read
                write_pos += samples_can_read
            else:
                buffer = self.sample_buffer[
                    self.start_sample : self.start_sample + samples
                ]
                output[write_pos : write_pos + samples] = buffer
                self.start_sample = self.start_sample + samples
                self.global_index += samples
                write_pos += samples
                samples = 0

        if samples != 0:
            output = output[:write_pos]
        
        return output

    def add_file_global_index(self, global_index):
        if len(self.file_global_indexes) - 1 < len(self.files):
            self.file_global_indexes.append(global_index)

    def look_up_file_and_index(self, index):
        file_index = None
        sample_index = None
        for i in range(len(self.file_global_indexes)):
            boundary = self.file_global_indexes[i]
            if index >= boundary:
                file_index = i
                sample_index = index - boundary

        if file_index != self.file_index:
            self.load_in_buf(file_index)
        self.start_sample = sample_index
        self.global_indexes = index
        self.finished_reading = False

    def gen_file_lookup_table(self):
        curr_index = self.file_global_indexes[0]
        for file in self.files:
            buf_file_name = self.signal_directory + file
            next_buf = self.load_func(buf_file_name)
            curr_index += len(next_buf)
            del next_buf
            self.add_file_global_index(curr_index)
        self.max_length = self.file_global_indexes[-1]

    def set_global_index(self, index):
        if index < self.max_length:
            self.look_up_file_and_index(index)
        else:
            self.finished_reading = True
            self.start_sample = self.max_length
            self.global_index = self.max_length

    def get_finished_reading(self):
        return self.finished_reading

    def get_id(self):
        return self.id

    def get_global_index(self):
        return self.global_index

    def reset(self):
        """
        Reset the reader to the start of the first file.
        """
        self.finished_reading = False
        self.curr_file_name = self.signal_directory + self.files[0]
        self.start_sample = 0
        self.global_index = 0
        self.file_index = 0
        del self.sample_buffer
        self.sample_buffer = self.load_func(self.curr_file_name)

    def sync(self, other_sf):
        """
        Synchronize this file reader with another, matching the current file and sample index.

        :param other_sf: Another Signal_File instance to synchronize with.
        """
        self.file_index = other_sf.file_index
        self.start_sample = other_sf.start_sample
        self.global_index = other_sf.global_index
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


class Event_File:
    def __init__(self, event_list, signal_file):
        self.events = event_list
        self.event_index = 0
        self.sf = signal_file
        if len(event_list) != 0:
            self.finished_reading = False
        else:
            self.finished_reading = True

    def get_events(self, num_events):
        event_signals = []
        events = []
        while not self.finished_reading and len(event_signals) < num_events:
            event = self.get_current_event()

            start_index = event[0]
            read_length = event[1] - event[0]
            self.sf.set_global_index(start_index)
            event_signal = self.sf.read(read_length)

            event_signals.append(event_signal)
            events.append(event)
            self.inc_event_index()

        return events, event_signals

    def get_current_event(self):
        if not self.finished_reading:
            out = self.events[self.event_index]
        else:
            out = self.events[-1]
        return out

    def inc_event_index(self):
        self.event_index += 1
        if self.event_index >= len(self.events):
            self.finished_reading = True

    def get_finished_reading(self):
        return self.finished_reading

    def get_id(self):
        return self.sf.get_id()

    def sync(self, other_ef):
        if not self.finished_reading and not other_ef.get_finished_reading():
            other_curr_event = other_ef.get_current_event()
            other_start = other_curr_event[0]

            curr_event = self.get_current_event()
            start = curr_event[0]

            if other_start < start:
                other_ef.sync(self)
            else:
                while other_start > start and not self.finished_reading:
                    self.inc_event_index()
                    curr_event = self.get_current_event()
                    start = curr_event[0]

    def reset(self):
        self.event_index = 0
        self.finished_reading = False
