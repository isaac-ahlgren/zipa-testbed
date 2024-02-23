import multiprocessing as mp
from multiprocessing import shared_memory

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
        self.buffer_ready = mp.Value("b", False)
        self.ready_buffer = shared_memory.SharedMemory(
            create=True, size=self.data_type.itemsize * buffer_size
        )
        self.buffer = None
        self.data_type = np.int32()

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            print("Currently inside of the mic's callback function.")
            self.ready_buffer = np.frombuffer(
                in_data, dtype=np.int32
            )  # Adjust for 16-bit data
            self.buffer_ready.value = True
            return (in_data, pyaudio.paContinue)

        return callback

    def start(self):
        # Start stream, recording for specified time interval
        self.stream.start_stream()

    def stop(self):
        self.buffer_ready.value = False
        self.stream.stop_stream()

    def extract(self):
        while True:
            if self.buffer_ready.value:
                break
        output = self.ready_buffer.copy()
        self.count += 1
        self.buffer_ready.value = False
        return output
