import time

import adafruit_sht31d
import board
import numpy as np
from typing import Any

from sensors.sensor_interface import SensorInterface


class SHT31D(SensorInterface):
    def __init__(self, config: dict[str, Any]) -> None:
        SensorInterface.__init__(self)
        self.sample_rate: int = config.get('sample_rate')
        self.buffer_size: int = config.get('sample_rate') * config.get('time_collected')
        self.chunk_size: int = config.get('chunk_size')
        self.chunks: int = int(self.buffer_size / self.chunk_size)
        self.name: str = "SHT31D"
        self.buffer: np.ndarray = np.zeros(self.chunk_size, np.float32())
        self.buffer_index: int = 0
        self.buffer_full: bool = False
        self.data_type: np.dtype = self.buffer.dtype
        self.sensor = adafruit_sht31d.SHT31D(board.I2C())

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
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
