import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


class Sensor_Reader:
    """
    Manages reading data from a sensor into a shared memory buffer, ensuring synchronized access
    among multiple client processes. This class initializes the shared memory, handles the polling
    of data from the sensor, and provides a mechanism to safely read the data.

    :param device: An instance of a sensor device that provides the data to be read.
    """
    def __init__(self, device):
        """
        Initializes the Sensor_Reader with a sensor device.

        :param device: The sensor object from which data will be read. Must have properties like data_type, buffer_size.
        """
        self.sensor = device
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=device.data_type.itemsize * device.buffer_size,
        )
        self.addressable_buffer = np.ndarray(
            (self.sensor.buffer_size,),
            dtype=self.sensor.data_type,
            buffer=self.shm.buf,
        )
        self.ready = mp.Value("b", False)
        self.pointer = mp.Value("i", 0)
        self.MAX_SENSOR_CLIENTS = 1024
        self.semaphore = mp.Semaphore(self.MAX_SENSOR_CLIENTS)
        self.mutex = mp.Semaphore()
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=device.name)
        self.poll_process.start()

    def poll(self):
        """
        Continuously polls data from the sensor and writes it into a shared memory buffer.
        Manages buffer wrapping and synchronization across processes accessing the data.
        """

        full_buffer = False
        self.mutex.acquire()
        # First pass when buffer isn't populated with sensor data
        while not full_buffer:
            data = self.sensor.extract()

            if self.pointer.value + len(data) >= self.sensor.buffer_size:
                full_buffer = True

            # Compensating for BMP280's 2D array
            if np.array(data).ndim > 1:
                data = data[0]

            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value = (self.pointer.value + 1) % self.sensor.buffer_size

        self.mutex.release()

        # After buffer is full
        while True:

            while self.semaphore.get_value() != self.MAX_SENSOR_CLIENTS:
                pass

            self.mutex.acquire()

            data = self.sensor.extract()

            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value = (self.pointer.value + 1) % self.sensor.buffer_size

            self.mutex.release()

    def read(self, sample_num):
        """
        Reads a specified number of samples from the shared memory buffer.

        :param sample_num: The number of samples to read.
        :return: An array of sensor data.
        :raises Exception: If the sample_num requested exceeds the buffer size.
        """
        if sample_num > self.sensor.buffer_size:
            raise Exception(
                "Sensor_Reader.read: Cannot request more data than the buffer size"
            )

        data = np.empty((sample_num,), dtype=self.sensor.data_type)

        while self.mutex.get_value() != 1:
            pass

        self.semaphore.acquire()

        for i in range(sample_num):
            data[i] = self.addressable_buffer[
                (self.pointer.value + i) % self.sensor.buffer_size
            ]
        self.semaphore.release()

        return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()

    def getter_thread(sen):
        length = 3 * 48000
        output = np.zeros(10 * length)
        for i in range(10):
            data = sen.read()
            for j in range(length):
                output[j + i * length] = data[j]
        plt.plot(output)
        plt.show()
        quit()

    from test_sensor import TestSensor

    ts = TestSensor(48000, 3 * 48000, signal_type="random")
    sen_reader = Sensor_Reader(ts)
    sen_thread(sen_reader)
    p = mp.Process(target=getter_thread, args=[sen_reader])
    p.start()
