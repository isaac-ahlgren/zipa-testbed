import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import pyaudio

from sensor_interface import SensorInterface
import time
import os
import wave

class Microphone(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        self.queue = mp.Queue()
        self.microphone_process = mp.Process(target=self.poll)

        self.format = pyaudio.paInt32  # Change to 16-bit format
        self.sampling = sample_rate
        self.name = "mic"
        self.pyaud = pyaudio.PyAudio()
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.stream = self.pyaud.open(
            format=self.format,
            channels=1,  # Stereo audio
            rate=sample_rate,
            input=True,
            frames_per_buffer=buffer_size,
        )
        self.data_type = np.int32()
        self.chunks_per_buffer = int(self.buffer_size / self.chunk_size)
        self.count = 0

        self.microphone_process.start()

    def poll(self):
        while True:
            output = self.stream.read(self.chunk_size)
            buf = np.frombuffer(
                output, dtype=np.int32
            )  # Adjust for 16-bit data
            self.queue.put(buf)

    def start(self):
        # Start stream, recording for specified time interval
        self.stream.start_stream()

    def stop(self):
        self.buffer_ready.value = False
        self.stream.stop_stream()

    def extract(self):
        return self.queue.get()
        

# Test case
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time
    import matplotlib.pyplot as plt
    mic = Microphone(44100, 44100*5, 1024) # Changed to 5; quicker
    sr = Sensor_Reader(mic)
    time.sleep(3)
    print("getting ready to read")
    results = sr.read(5*44100)
    # plt.plot(sr.read(5*44100)) # Changed to 5; quicker
    print("results collected, saving file") 
    filename = "recording.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(mic.pyaud.get_sample_size(pyaudio.paInt32))
        wf.setframerate(mic.sampling)
        wf.writeframes(b"".join(results))
    print(f"Audio file written. Size: {os.path.getsize(filename)} bytes")
    plt.show()  
