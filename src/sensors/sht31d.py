import time

import adafruit_sht31d
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class SHT31D(SensorInterface):
    """
    A sensor interface for the SHT31D, a high-accuracy temperature and humidity sensor. This class manages
    reading humidity data from the sensor and buffers it for consumption.

    :param config: Configuration dictionary that includes sample rate, time collected, and chunk size.
    """
    def __init__(self, config):
        """
        Initializes the SHT31D sensor interface with configuration settings.

        :param config: A dictionary with configuration settings such as sample rate, time collected,
                       and chunk size for data reading.
        """
        SensorInterface.__init__(self)
        self.sample_rate = config.get('sample_rate')
        self.buffer_size = config.get('sample_rate') * config.get('time_collected')
        self.chunk_size = config.get('chunk_size')
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "SHT31D"
        self.buffer = np.zeros(self.chunk_size, np.float32())
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.sensor = adafruit_sht31d.SHT31D(board.I2C())

    def start(self):
        """
        Start the sensor (placeholder for actual start commands if necessary).
        """
        pass

    def stop(self):
        """
        Stop the sensor (placeholder for actual stop commands if necessary).
        """
        pass

    def read(self):
        """
        Reads humidity data from the sensor and returns it as a NumPy array.

        :return: A NumPy array containing humidity readings from the sensor.
        """
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
