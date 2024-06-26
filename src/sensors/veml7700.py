import time

import adafruit_veml7700
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class VEML7700(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        SensorInterface.__init__(self)
        self.name = "lux"
        # Sensor configuration parameters
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.buffer = np.zeros(
            chunk_size, np.float32()
        )  # Initialize buffer for lux readings
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.light = adafruit_veml7700.VEML7700(board.I2C())
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        data = np.empty(self.chunk_size, self.data_type)

        for i in range(self.chunk_size):
            lux = self.light.lux
            data[i] = np.float32(lux)
            time.sleep(1 / self.sample_rate)

        return data


if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    veml = VEML7700(40, 40 * 5, 8)
    sr = Sensor_Reader(veml)

    time.sleep(3)
    print("Beginning reading.")

    for i in range(10):
        results = sr.read(40 * 5)
        print(f"Number of results: {len(results)},\n {results}")
        time.sleep(10)

    exit()
