import multiprocessing as mp
from multiprocessing import shared_memory
from microphone import Microphone

import numpy as np

MAX_CLIENTS = 0x400

# MAIN TEST FOR THIS:
# you want multiple processes (threads) to read from the buffer at the same time during
class Sensor():
    def __init__(self, device):
        self.sensor = device
        self.shm = shared_memory.SharedMemory(
            create=True,
            size=device.data_type.itemsize * device.buffer_size,
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
        self.mutex = mp.Lock()
        self.sensor.start()

    def poll(self):
        while True:
            while self.semaphore._Semaphore__value != self.MAX_SENSOR_CLIENTS:
                pass

            self.mutex.acquire()

            data = self.sensor.extract()

            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value += 1 % self.sensor.buffer_size

    def read(self):
        data = np.empty((self.sensor.buffer_size,), dtype=self.sensor.data_type)
        
        print("Checking if mutex lock is still in play...")
        while self.mutex.locked(): 
            pass

        self.semaphore.acquire()

        for d in range(self.sensor.buffer_size):
            data[d] = self.addressable_buffer[
                (self.pointer.value + d) % self.sensor.buffer_size
            ]
        
        self.semaphore.release()

        return data
    


    if __name__ == "__main__":
        from test_sensor import Test_Sensor
        ts = Test_Sensor(48000, 3*48000, signal_type='sine')
        sen_reader = Sensor(ts)
