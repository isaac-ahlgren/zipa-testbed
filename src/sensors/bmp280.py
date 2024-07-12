import time

import adafruit_bmp280
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class BMP280(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        # Sensor configuration parameters
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size * 3  # Returns Temp, Altitude, and Pressure
        self.chunk_size = chunk_size * 3
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "bmp280"
        self.data_type = np.float32()
        self.buffer = np.empty(
            (chunk_size,), dtype=np.float32()
        )  # Initialize buffer for temperature, pressure, altitude
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.bmp = adafruit_bmp280.Adafruit_BMP280_I2C(board.I2C())

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        # rows x columns for pandas readibility
        data = np.empty((self.chunk_size, 3), self.data_type)

        for i in range(self.chunk_size):
            data[i, 0] = self.bmp.temperature
            data[i, 1] = self.bmp.pressure
            data[i, 2] = self.bmp.altitude

        return data



if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    bmp = BMP280(50, 50, 25)
    sr = Sensor_Reader(bmp)

    time.sleep(3)  # BMP needs ten seconds to populate on 1st pass
    print("Getting ready to read.")

    for i in range(5):
        results = sr.read(50 * 3)
        print(results)

        time.sleep(5)

    exit()
