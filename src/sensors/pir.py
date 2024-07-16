# import multiprocessing
import time
from typing import Any, Dict

import numpy as np
import RPi.GPIO as GPIO

from sensors.sensor_interface import SensorInterface


class PIR(SensorInterface):
    """
    A sensor interface for a Passive Infrared (PIR) sensor, used to detect motion via infrared light.
    This class configures a GPIO pin for input and reads motion data from the PIR sensor.

    :param config: Dictionary containing configuration parameters such as sample rate, time collected, and chunk size.
    :param pin: GPIO pin number to which the PIR sensor is connected (defaults to 12).
    """

    def __init__(self, config: Dict[str, Any], pin: int = 12) -> None:
        SensorInterface.__init__(self)
        self.sample_rate = config.get("sample_rate")
        self.buffer_size = config.get("sample_rate") * config.get("time_collected")
        self.chunk_size = config.get("chunk_size")
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "PIR"
        self.pin = pin
        self.buffer = np.zeros(self.buffer_size, dtype=int)  # Store states as 1s and 0s
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)

    def start(self) -> None:
        """
        Start the sensor or any necessary configurations before reading data.
        Placeholder for future implementation if required.
        """
        pass

    def stop(self) -> None:
        """
        Stop the sensor or clean up configurations after reading data.
        Placeholder for future implementation if required.
        """
        pass

    def read(self) -> np.ndarray:
        """
        Reads data from the PIR sensor. Captures motion detection as a series of binary states.

        :return: Numpy array containing a chunk of PIR sensor readings.
        """
        data = np.empty(self.chunk_size, self.data_type)

        for i in range(self.chunk_size):
            data[i] = GPIO.input(self.pin)
            time.sleep(1 / self.sample_rate)

        return data


if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    pir = PIR(2, 10, 2)
    sr = Sensor_Reader(pir)

    time.sleep(3)
    print("Getting ready to read.")

    for i in range(5):
        results = sr.read(10)
        print(results)
        time.sleep(10)

    exit()
