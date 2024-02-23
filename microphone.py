import time

import numpy as np
import pyaudio

from sensor_interface import SensorInterface


# TODO: Make template class for all sensors so that they all have the same functions
class Microphone(SensorInterface):
    def __init__(self, sample_rate, buffer_size):
        self.format = pyaudio.paInt32  # Change to 16-bit format
        self.sampling = sample_rate
        self.name = "mic"
        self.pyaud = pyaudio.PyAudio()
        self.buffer_size = buffer_size
        self.chunk_size = buffer_size
        self.stream = self.pyaud.open(
            format=self.format,
            channels=1,  # Stereo audio
            rate=sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.get_callback(),
        )
        self.chunks_per_buffer = int(self.buffer_size / self.chunk_size)
        self.count = 0
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None
        self.data_type = np.int32()

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.ready_buffer = np.frombuffer(
                in_data, dtype=np.int32
            )  # Adjust for 16-bit data
            self.buffer_ready = True
            return (in_data, pyaudio.paContinue)

        return callback

    def start(self):
        # Start stream, recording for specified time interval
        self.stream.start_stream()

    def stop(self):
        self.buffer_ready = False
        self.stream.stop_stream()

    def extract(self):
        while True:
            if self.buffer_ready:
                break
        self.buffer = self.ready_buffer
        self.count += 1
        self.buffer_ready = False
        return self.buffer
