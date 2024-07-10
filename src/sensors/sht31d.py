import time

import adafruit_sht31d
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class SHT31D(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "sht31d"
        self.buffer = np.zeros(chunk_size, np.float32())
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.sensor = adafruit_sht31d.SHT31D(board.I2C())
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        data = np.empty(self.chunk_size, self.data_type)

        for i in range(self.chunk_size):
            humidity = self.sensor.relative_humidity
            data[i] = np.float32(humidity)
            time.sleep(1 / self.sample_rate)

        return data


if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    sht31d = SHT31D(40, 40 * 5, 8)
    sr = Sensor_Reader(sht31d)

    time.sleep(3)
    print("Beginning reading.")

    for i in range(10):
        results = sr.read(40 * 5)
        print(f"Number of results: {len(results)},\n {results}")
        time.sleep(10)

    exit()
