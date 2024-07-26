import time
from typing import Any

import numpy as np
from sensor_interface import SensorInterface


class TestSensor(SensorInterface):
    """
    A test sensor class that simulates sensor data for testing purposes. It can generate data according to a specified
    signal type, either as a sine wave or random noise, at a specified sample rate, buffer size, and chunk size.

    :param config: Configuration dictionary with keys 'sample_rate', 'time_collected', and 'chunk_size'.
    :param signal_type: Type of signal to generate ('sine' for sine wave, 'random' for random noise). Default is 'sine'.
    """

    __test__ = False

    def __init__(self, config: dict[str, Any], signal_type: str = "sine") -> None:
        """
        Initializes the TestSensor with given configurations and signal type.

        :param config: A dictionary containing configuration parameters such as sample rate, time collected, and chunk size.
        :param signal_type: The type of signal this sensor should generate ('sine' or 'random').
        """
        SensorInterface.__init__(self)
        self.sample_rate: int = config.get("sample_rate")
        self.buffer_size: int = config.get("sample_rate") * config.get("time_collected")
        self.chunk_size: int = config.get("chunk_size")
        self.name: str = "test_sensor"
        self.time: int = 0
        self.buffer_ready: bool = False
        self.ready_buffer: np.ndarray = None
        self.buffer: np.ndarray = None
        self.data_type: np.dtype = np.float32()
        self.data_type_size: int = 4
        self.signal_type: str = signal_type

    def start(self) -> None:
        """
        Start the sensor operation (not implemented for test purposes).
        """
        pass

    def stop(self) -> None:
        """
        Stop the sensor operation (not implemented for test purposes).
        """
        pass

    def read(self) -> np.ndarray:
        """
        Simulates reading data from the sensor. Generates data according to the specified signal type.

        :return: A numpy array containing the simulated sensor data.
        """
        time.sleep(self.chunk_size / self.sample_rate)
        output = np.zeros(self.chunk_size, dtype=self.data_type)

        if self.signal_type == "random":
            rng = np.random.default_rng()
            for i in range(len(output)):
                output[i] = rng.random()
        elif self.signal_type == "sine":
            for i in range(len(output)):
                output[i] = np.sin(2 * np.pi / self.sample_rate * (i + self.time))
            self.time += len(output)

        return output
