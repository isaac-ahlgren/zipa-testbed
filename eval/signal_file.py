import glob
from typing import Callable

import numpy as np

class Signal_File:
    def __init__(
        self,
        signal_directory: str,
        file_names: str,
        wrap_around_read: bool = False,
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
        self.sample_buffer = self.load_func(self.signal_directory + self.files[0])
        self.wrap_around_read = wrap_around_read
        self.finished_reading = False
        self.id = id

    def switch_files(self):
        """
        Switch to the next file in the directory or wrap around if enabled.
        """
        self.start_sample = 0
        self.file_index += 1
        print("Loading in " + self.signal_directory + self.files[self.file_index])
        if len(self.files) == self.file_index and self.wrap_around_read: # Reset if wrap around enabled
            self.reset()
        else:  
            del self.sample_buffer
            if len(self.files) == self.file_index: # If no more to read, set the finished reading flag
                self.finished_reading = False
            else:
                self.sample_buffer = self.load_func(
                    self.signal_directory + self.files[self.file_index]
                )

    def read(self, samples: int) -> np.ndarray:
        """
        Read a specified number of samples across multiple files.

        :param samples: Number of samples to read.
        :return: Array containing the read samples.
        """
        output = np.array([])

        while samples != 0 and not self.finished_reading :
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
        self.start_sample = 0
        self.file_index = 0
        self.finished_reading = True
        del self.sample_buffer
        self.sample_buffer = self.load_func(self.signal_directory + self.files[0])

    def sync(self, other_sf: "Signal_File"):
        """
        Synchronize this file reader with another, matching the current file and sample index.

        :param other_sf: Another Signal_File instance to synchronize with.
        """
        self.file_index = other_sf.file_index
        self.start_sample = other_sf.start_sample
        del self.sample_buffer
        self.sample_buffer = self.load_func(
            self.signal_directory + self.files[self.file_index]
        )