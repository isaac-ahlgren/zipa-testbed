import time

import adafruit_veml7700
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class VEML7700(SensorInterface):
    """
    A sensor interface for the VEML7700, a high accuracy ambient light sensor. This class manages
    the configuration and data collection from the VEML7700 sensor, storing lux readings in a buffer.

    :param config: Configuration dictionary that includes sample rate, time collected, and chunk size.
    """
    def __init__(self, config):
        """
        Initializes the VEML7700 sensor interface with configuration settings.

        :param config: A dictionary containing configuration parameters such as sample rate, time collected,
                       and chunk size for data reading.
        """
        SensorInterface.__init__(self)
        self.name = "VEML7700"
        # Sensor configuration parameters
        self.sample_rate = config.get('sample_rate')
        self.buffer_size = config.get('sample_rate') * config.get('time_collected')
        self.chunk_size = config.get('chunk_size')
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.buffer = np.zeros(
            self.chunk_size, np.float32()
        )  # Initialize buffer for lux readings
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.light = adafruit_veml7700.VEML7700(board.I2C())

    def start(self):
        """
        Start the sensor (not implemented as VEML7700 has no specific start requirements).
        """
        pass

    def stop(self):
        """
        Stop the sensor (not implemented as VEML7700 has no specific stop requirements).
        """
        pass

    def read(self):
        """
        Reads lux data from the VEML7700 sensor and returns it as a NumPy array.

        :return: A NumPy array containing lux readings from the sensor.
        """
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
