import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List

import numpy as np

class Sensor:
    """
    A sensor manager class that handles reading data from a sensor device into a shared memory space
    for concurrent access by multiple processes. This class uses semaphores and mutexes to manage
    access and ensure data integrity.

    :param device: An instance of a device with attributes like sample_rate, name, buffer_size, and data_type.
    """
    def __init__(self, device: 'DeviceInterface') -> None:
        """
        Initializes the Sensor object with a given device.

        :param device: The device object which must have properties like sample_rate, name, buffer_size, and data_type.
        """
        self.sensor = device
        self.sample_rate = device.sample_rate
        self.name = device.name
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=device.data_type_size * device.buffer_size,
            name="sensor_buffer",
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

    def poll(self) -> None:
        """
        Continuously polls data from the sensor device and stores it in shared memory.
        """
        while True:
            while self.semaphore.get_value() != self.MAX_SENSOR_CLIENTS:
                pass

            self.mutex.acquire()

            data = self.sensor.extract()

            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value = (self.pointer.value + 1) % self.sensor.buffer_size
            self.mutex.release()

    def read(self) -> np.ndarray:
        """
        Reads data from the shared memory where the sensor data is stored.

        :return: A numpy array containing the buffered sensor data.
        """
        data = np.empty((self.sensor.buffer_size,), dtype=self.sensor.data_type)

        print("Checking if mutex lock is still in play...")
        while self.mutex.get_value() != 1:
            pass

        self.semaphore.acquire()

        for d in range(self.sensor.buffer_size):
            data[d] = self.addressable_buffer[
                (self.pointer.value + d) % self.sensor.buffer_size
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

    ts = TestSensor(48000, 3 * 48000, signal_type="sine")
    sen_reader = Sensor(ts)
    sen_thread(sen_reader)
    p = mp.Process(target=getter_thread, args=[sen_reader])
    p.start()
