import multiprocessing
import time

import numpy as np
import RPi.GPIO as GPIO

from sensors.sensor_interface import SensorInterface


class PIR(SensorInterface):
    def __init__(self, config, pin=12):
        SensorInterface.__init__(self)
        self.sample_rate = config.get('sample_rate')
        self.buffer_size = config.get('sample_rate') * config.get('time_collected')
        self.chunk_size = config.get('chunk_size')
        self.chunks = int(self.buffer_size / self.chunk_size)
        self.name = "pir"
        self.pin = pin
        self.buffer = np.zeros(self.buffer_size, dtype=int)  # Store states as 1s and 0s
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
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
