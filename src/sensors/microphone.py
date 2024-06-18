import multiprocessing as mp
import os
import time
import wave
from multiprocessing import shared_memory

import numpy as np
import pyaudio

from sensors.sensor_interface import SensorInterface

class Microphone(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size, rms_filter_enabled=False):
        # When the RMS filter is enabled, the true sampling rate will be sample_rate/chunk_size.
        # Each chunk size will be converted into one sample by performing RMS on the chunk.
        SensorInterface.__init__(self)
        
        self.rms_filter_enabled = rms_filter_enabled

        self.format = pyaudio.paInt32 
        self.sample_rate = sample_rate
        self.name = "mic"
        self.pyaud = pyaudio.PyAudio()
        self.chunk_size = chunk_size
        self.samples_per_spl_sample = sample_rate // chunk_size
 
        # This is to account for the differences in buffer size between performing the RMS filter and not performing it
        self.buffer_size = buffer_size
        if rms_filter_enabled:
            self.buffer_size = buffer_size * self.samples_per_spl_sample

        self.stream = self.pyaud.open(
            format=self.format,
            channels=1,  # Stereo audio
            rate=sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
        )

        # Data types will switch between using RMS filter or not, mostly because we have to be able to fit all the RMS data into a single datatype
        self.data_type = np.int32()
        if rms_filter_enabled:
            self.data_type = np.float64()

        self.start_thread()

    def start(self):
        # Start stream, recording for specified time interval
        self.stream.start_stream()

    def stop(self):
        self.buffer_ready.value = False
        self.stream.stop_stream()

    def calc_rms(self, signal):
        output = np.zeros(self.chunk_size, dtype=np.float64)
        for i in range(self.chunk_size):
            arr = signal[
                i * self.samples_per_spl_sample : (i + 1) * self.samples_per_spl_sample
            ].astype(np.int64)
            output[i] = np.sqrt(np.mean(arr**2)) / self.samples_per_spl_sample
        return output

    def read(self):
        output = self.stream.read(self.samples_per_spl_sample * self.chunk_size)
        buf = np.frombuffer(output, dtype=np.int32)
        if self.rms_filter_enabled:
            buf = self.filter(buf)
        return buf

# Test case
if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    mic = Microphone(120, 120 * 5, 120)
    sen_reader = Sensor_Reader(mic)
    time.sleep(3)
    print(sen_reader.read(240))
