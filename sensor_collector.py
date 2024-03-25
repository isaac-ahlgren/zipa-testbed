import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np

from nfs import NFSLogger


# MAIN TEST FOR THIS:
# you want multiple processes (threads) to read from the buffer at the same time during
class SensorCollector:
    def __init__(self, device, logger):
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
        self.logger = logger
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=device.name)
        self.poll_process.start()

    def poll(self):
        full_buffer = False
        self.mutex.acquire()
        # First pass when buffer isn't populated with sensor data
        while not full_buffer:
            data = self.sensor.extract()

            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value = (self.pointer.value + 1) % self.sensor.buffer_size

            if self.pointer.value + len(data) >= self.sensor.buffer_size:
                full_buffer = True

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

    # TODO Redo this to route chunks to NFS server.
    def collect(self, sample_num):
        """
        Create file for sensor if not in NFS; append chunks to file if exists

        CSV File

        Make sure you're mounted to the NFS

        Log if file was created using NFS logger
        """
        if sample_num > self.sensor.buffer_size:
            raise Exception(
                "Sensor_Reader.read: Cannot request more data than the buffer size"
            )

        data = np.empty((sample_num,), dtype=self.sensor.data_type)

        while self.mutex.get_value() != 1:
            pass

        self.semaphore.acquire()

        for d in range(sample_num):
            data[d] = self.addressable_buffer[
                (self.pointer.value + d) % self.sensor.buffer_size
            ]

        self.semaphore.release()

        csv = ", ".join(str(num) for num in data)
        self.logger.log((str(self.sensor.name), "csv", csv))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()

    def getter_thread(sen):
        length = 3 * 48000
        output = np.zeros(10 * length)
        for i in range(10):
            sen.collect(40_000)
            # for j in range(length):
            #     output[j + i * length] = data[j]
        # plt.plot(output)
        # plt.show()
        quit()

    from test_sensor import Test_Sensor

    ts = Test_Sensor(48000, 3 * 48000, 12_000, signal_type="random")
    sen_reader = SensorCollector(ts, NFSLogger(
                      user='luke',
                      password='lucor011&',
                      host='10.17.29.18',
                      database='file_log',
                      nfs_server_dir='/mnt/data',  # Make sure this directory exists and is writable
                      identifier='192.168.1.220'  # Could be IP address or any unique identifier
                    ))
    sen_thread(sen_reader)
    p = mp.Process(target=getter_thread, args=[sen_reader])
    p.start()
