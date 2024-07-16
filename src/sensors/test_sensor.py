import time

import numpy as np
from typing import Any

from sensors.sensor_interface import SensorInterface


class TestSensor(SensorInterface):
    def __init__(self, config: dict[str, Any], signal_type: str = "sine") -> None:
        SensorInterface.__init__(self)
        self.sample_rate: int = config.get('sample_rate')
        self.buffer_size: int = config.get('sample_rate') *config.get('time_collected')
        self.chunk_size: int = config.get('chunk_size')
        self.name: str = "test_sensor"
        self.time: int = 0
        self.buffer_ready: bool = False
        self.ready_buffer: np.ndarray = None
        self.buffer: np.ndarray = None
        self.data_type: np.dtype = np.float32()
        self.data_type_size: int = 4
        self.signal_type: str = signal_type

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> np.ndarray:
        time.sleep(self.chunk_size / self.sample_rate)
        output = np.zeros(self.chunk_size, dtype=self.data_type)

        if self.signal_type == "random":
            rng = np.random.default_rng()
            for i in range(len(output)):
                output[i] = rng.random()
        elif self.signal_type == "sine":
            for i in range(len(output)):
                output[i] = np.sin(2 * np.pi / self.sample_rate * i)
            self.time += len(output)

        return output
