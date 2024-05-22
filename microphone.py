import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import pyaudio

from sensor_interface import SensorInterface
import time
import os
import wave

class Microphone(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size, mic_sampling_rate=48000):
        SensorInterface.__init__(self)
        self.format = pyaudio.paInt32  # Change to 16-bit format
        self.sample_rate = sample_rate
        self.name = "mic"
        self.pyaud = pyaudio.PyAudio()
        self.chunk_size = chunk_size
        self.samples_per_spl_sample = mic_sampling_rate // sample_rate
        self.buffer_size = buffer_size*self.samples_per_spl_sample
        self.stream = self.pyaud.open(
            format=self.format,
            channels=1,  # Stereo audio
            rate=mic_sampling_rate,
            input=True,
            frames_per_buffer=self.buffer_size
        )
        self.data_type = np.int32()
        self.count = 0
        self.start_thread()


    def start(self):
        # Start stream, recording for specified time interval
        self.stream.start_stream()

    def stop(self):
        self.buffer_ready.value = False
        self.stream.stop_stream()

    def calc_rms(self, signal):
        output = np.zeros(self.chunk_size, dtype=np.float32)
        for i in range(self.chunk_size):
            output[i] = np.sqrt(np.mean(signal[i*self.samples_per_spl_sample:(i+1)*self.samples_per_spl_sample]**2)) / len(signal)
        rms = np.sqrt(np.mean(signal**2)) / len(signal)
        return rms

    def read(self):
        output = self.stream.read(self.samples_per_spl_sample*self.chunk_size)
        buf = np.frombuffer(
            output, dtype=np.int32
        )
        return self.calc_rms(buf)
        

# Test case
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
  
    mic = Microphone(120, 1000000, 120) 
    sen_reader = Sensor_Reader(mic)
    print(sen_reader.read(240))