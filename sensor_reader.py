import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


# MAIN TEST FOR THIS:
# you want multiple processes (threads) to read from the buffer at the same time during
class Sensor_Reader:
    def __init__(self, device):
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
        print("Beginning polling process.")
        while True:

            while self.semaphore.get_value() != self.MAX_SENSOR_CLIENTS:
                pass

            self.mutex.acquire()
            data = self.sensor.extract()
            print(f"Recieved from: {self.sensor.name} --> {data}")
            quit()
            print("Data extraction complete.")
            for d in data:
                self.addressable_buffer[self.pointer.value] = d
                self.pointer.value = (self.pointer.value + 1) % self.sensor.buffer_size
            self.mutex.release()

    def read(self):
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
        length = 3*48000
        output = np.zeros(10*length)
        for i in range(10):
            data = sen.read()
            for j in range(length):
                output[j + i*length] = data[j]
        plt.plot(output)
        plt.show()
        quit()

    from test_sensor import Test_Sensor
    ts = Test_Sensor(48000, 3*48000, signal_type='random')
    sen_reader = Sensor_Reader(ts)
    sen_thread(sen_reader)
    p = mp.Process(target=getter_thread, args=[sen_reader])
    p.start()

