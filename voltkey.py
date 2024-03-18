import numpy as np
import serial

from sensor_interface import SensorInterface

SETUP      = "s".encode()
ACCEPT     = "a".encode()
CANCEL     = "x".encode()
GO         = "g".encode()
HARD_RESET = "h".encode()
RESET      = "r".encode()


class Voltkey(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.chunks = self.buffer_size // self.chunk_size
        self.name = "voltkey"
        self.buffer = np.zeros(chunk_size, np.float32())
        self.buffer_index = 0
        self.data_type = self.buffer.dtype
        self.sensor = serial.Serial("/dev/ttyS0")
        self.sensor.baudrate = 9600  # TODO Verify if a different rate is needed
        self.start_thread()

    def start(self):
        self.sensor.write(SETUP)

        for char in str(self.chunk_size):
            self.sensor.writable(char.encode())

        self.sensor.write(ACCEPT)

    def stop(self):
        self.sensor.write(CANCEL)

    def read(self):
        # TODO Revisit this with Max
        data = np.empty(self.chunk_size, self.data_type)
        signal = self.sensor.read(self.chunk_size)

        for i in range(self.chunk_size):
            data[i] = signal[i]

        return data


"""
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time

    voltkey = Voltkey() # TODO populate arguments
    sr = Sensor_Reader(voltkey)
    time.sleep(3)
    print("Beginning reading.")

    for i in range(10):
        results = sr.read() # TODO populate arguments
        print(f"Number of results: {len(results)},\n{results}")
        time.sleep(10)

    exit()
"""
