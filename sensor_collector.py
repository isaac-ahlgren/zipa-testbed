import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import shared_memory

import numpy as np

from nfs import NFSLogger


class SensorCollector:
    """
    Class copies `Sensor_Reader`, but refitted to collect and log sensor data
    onto NFS server and MySQL. Noticable difference is the absence of `read`
    function, being replaced by `collect`.
    """

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

    def collect(self, sample_num):
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
    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()

    def getter_thread(sen):
        for i in range(10):
            sen.collect(40_000)

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
