import time

import numpy as np

from sensors.sensor_interface import SensorInterface


class Test_Sensor(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size, signal_type="sine"):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.name = "test_sensor"
        self.time = 0
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None
        self.data_type = np.float32()
        self.data_type_size = 4
        self.signal_type = signal_type
        self.start_thread()

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
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
