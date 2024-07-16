import time

import numpy as np

from sensors.sensor_interface import SensorInterface


class TestSensor(SensorInterface):
    """
    A test sensor class that simulates sensor data for testing purposes. It can generate data according to a specified
    signal type, either as a sine wave or random noise, at a specified sample rate, buffer size, and chunk size.

    :param config: Configuration dictionary with keys 'sample_rate', 'time_collected', and 'chunk_size'.
    :param signal_type: Type of signal to generate ('sine' for sine wave, 'random' for random noise). Default is 'sine'.
    """
    def __init__(self, config, signal_type="sine"):
        """
        Initializes the TestSensor with given configurations and signal type.

        :param config: A dictionary containing configuration parameters such as sample rate, time collected, and chunk size.
        :param signal_type: The type of signal this sensor should generate ('sine' or 'random').
        """
        SensorInterface.__init__(self)
        self.sample_rate = config.get('sample_rate')
        self.buffer_size = config.get('sample_rate') *config.get('time_collected')
        self.chunk_size = config.get('chunk_size')
        self.name = "test_sensor"
        self.time = 0
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None
        self.data_type = np.float32()
        self.data_type_size = 4
        self.signal_type = signal_type

    def start(self):
        """
        Start the sensor operation (not implemented for test purposes).
        """
        pass

    def stop(self):
        """
        Stop the sensor operation (not implemented for test purposes).
        """
        pass

    def read(self):
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
                output[i] = np.sin(2 * np.pi / self.sample_rate * i)
            self.time += len(output)

        return output
