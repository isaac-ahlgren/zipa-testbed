import time

import adafruit_ahtx0
import board
import numpy as np

from sensor_interface import SensorInterface

class AHT20(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size * 2 # Returns temp and humidity
        self.chunk_size = chunk_size * 2
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "aht20"
        self.buffer = np.zeros(chunk_size, np.float32())
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.sensor = adafruit_ahtx0.AHTx0(board.I2C())
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        data = np.empty(self.chunk_size, self.data_type)

        for i in range(0, self.chunk_size, 2):
            data[i] = self.sensor.temperature
            data[i + 1] = self.sensor.relative_humidity
            time.sleep(1 / self.sample_rate)

        return data
    
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time
    
    aht20 = AHT20(50, 50, 25)
    sr = Sensor_Reader(aht20)
    
    time.sleep(3)
    print("Beginning reading.")

    for i in range(10):
        results = sr.read(50 * 2)
        print(f"Number of results: {len(results)},\n {results}")
        time.sleep(5)

    exit()