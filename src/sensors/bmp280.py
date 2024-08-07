import time
from typing import Any, Dict

import adafruit_bmp280
import board
import numpy as np

from sensors.sensor_interface import SensorInterface


class BMP280(SensorInterface):
    """
    A sensor interface for the BMP280 sensor, which measures temperature, pressure, and altitude.

    :param config: Configuration dictionary that includes sample rate, time collected, and chunk size.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Sensor configuration parameters
        SensorInterface.__init__(self)
        self.sample_rate = config.get("sample_rate")
        self.buffer_size = (
            config.get("sample_rate") * config.get("time_collected") * 3
        )  # Returns Temp, Altitude, and Pressure
        self.chunk_size = config.get("chunk_size") * 3

        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "BMP280"
        self.data_type = np.float32()
        self.buffer = np.empty(
            (self.chunk_size,), dtype=np.float32()
        )  # Initialize buffer for temperature, pressure, altitude
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        self.bmp = adafruit_bmp280.Adafruit_BMP280_I2C(board.I2C())

    def start(self) -> None:
        """
        Start the sensor (method to be implemented if sensor requires activation).
        """
        pass

    def stop(self) -> None:
        """
        Stop the sensor (method to be implemented if sensor requires deactivation).
        """
        pass

    def read(self) -> np.ndarray:
        """
        Reads data from the BMP280 sensor. This method collects temperature, pressure, and altitude data.

        :return: A numpy array containing the sensor readings formatted as [temperature, pressure, altitude].
        """
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
