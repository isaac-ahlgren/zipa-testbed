import multiprocessing
import time
import board
import adafruit_bmp280
from sensor_interface import SensorInterface
import numpy as np

class BMP280Sensor(SensorInterface):
    def __init__(self, sample_rate, buffer_size):
        # Sensor configuration parameters
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=self._sample_sensor, args=(self.child_conn,))
        self.buffer = np.zeros((buffer_size, 3))  # Initialize buffer for temperature, pressure, altitude
        self.buffer_index = 0
        self.buffer_full = False

    def _sample_sensor(self, conn):
        # Initialize the sensor in the child process
        i2c = board.I2C()
        bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)

        while True:
            if conn.poll():  # Check for the stop signal
                if conn.recv() == 'STOP':
                    break

            temperature = bmp280.temperature
            pressure = bmp280.pressure
            altitude = bmp280.altitude

            # Send the data back to the parent process
            conn.send([temperature, pressure, altitude])
            time.sleep(1 / self.sample_rate)

    def start(self):
        self.process.start()

    def stop(self):
        self.parent_conn.send('STOP')
        self.process.join()

    def extract(self):
        while self.parent_conn.poll():  # Check if there's data to read
            data = self.parent_conn.recv()  # Read data
            self.buffer[self.buffer_index] = data
            self.buffer_index += 1
            if self.buffer_index >= self.buffer_size:
                self.buffer_index = 0
                self.buffer_full = True
        
        # Return a copy of the buffer
        if self.buffer_full:
            return self.buffer.copy()
        else:
            return self.buffer[:self.buffer_index].copy()
