import time

import adafruit_lps2x
import board
import numpy as np

from sensor_interface import SensorInterface

class LPS22(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "LPS22"
        self.buffer = np.zeros(chunk_size, np.float32())
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.sensor = adafruit_lps2x.LPS22(board.I2C())
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        data = np.empty(self.chunk_size, self.data_type)

        for i in range(self.chunk_size):
            pressure = self.sensor.pressure
            data[i] = np.float32(pressure)
            time.sleep(1 / self.sample_rate)

        return data
    
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time
    lps22 = LPS22(40, 40 * 5, 8)
    sr = Sensor_Reader(lps22)
    time.sleep(3)
    print("Beginning reading.")
    for i in range(10):
        results = sr.read(40 * 5)
        print(f"Number of results: {len(results)},\n {results}")
        time.sleep(10)
    exit()