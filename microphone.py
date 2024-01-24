import wave

import numpy as np
import pyaudio


class Microphone:
    def __init__(self, sample_rate, buffer_size):
        self.format = pyaudio.paFloat32
        self.sample_rate = sample_rate
        self.pyaud = pyaudio.PyAudio()
        self.buffer_size = buffer_size
        self.chunk_size = buffer_size
        # Opens an audio stream for input
        self.stream = self.pyaud.open(
            format=self.format,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Stores the data in a buffer to be used in the protocol
            stream_callback=self.get_callback(),
        )
        self.chunks_per_buffer = int(buffer_size / self.chunk_size)
        self.count = 0
        # Initialize but keep instance on standby
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            # Store data in a buffer
            self.ready_buffer = np.fromstring(in_data, dtype=np.float32)
            # The buffer is available with new data
            self.buffer_ready = True
            # Continue running the audio stream
            return (in_data, pyaudio.paContinue)

        return callback

    def start_stream(self):
        self.stream.start_stream()

    def stop_stream(self):
        self.buffer_ready = False
        self.stream.stop_stream()

    def get_audio(self):
        # Wait for audio collection until the buffer is marked ready
        while 1:
            if self.buffer_ready:
                break
        # Populate the buffer with the last 3/4 of collected data
        self.buffer = self.ready_buffer[int(len(self.ready_buffer) / 4) :]
        # Keep track of how many times audio has been collected
        self.count += 1
        # Buffer is full of data
        self.buffer_ready = False
        return self.buffer
