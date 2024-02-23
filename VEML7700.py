import multiprocessing
import time
import board
import adafruit_veml7700
from sensor_interface import SensorInterface
import numpy as np

class LightSensor(SensorInterface):
    def __init__(self, sample_rate, buffer_size):
        # Sensor configuration parameters
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=self._read_sensor, args=(self.child_conn,))
        self.buffer = np.zeros(buffer_size)  # Initialize buffer for lux readings
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype

    def _read_sensor(self, conn):
        # Initialize the sensor in the child process
        i2c = board.I2C()
        sensor = adafruit_veml7700.VEML7700(i2c)

        while True:
            if conn.poll():  # Check for the stop signal
                if conn.recv() == 'STOP':
                    break
            
            lux = sensor.lux
            # Send the lux data back to the parent process
            conn.send(lux)
            time.sleep(1 / self.sample_rate)

    def start(self):
        self.process.start()

    def stop(self):
        self.parent_conn.send('STOP')
        self.process.join()

    def extract(self):
        while self.parent_conn.poll():  # Check if there's data to read
            lux = self.parent_conn.recv()  # Read lux data
            self.buffer[self.buffer_index] = lux
            self.buffer_index += 1
            if self.buffer_index >= self.buffer_size:
                self.buffer_index = 0
                self.buffer_full = True

        # Return a copy of the buffer
        if self.buffer_full:
            return self.buffer.copy()
        else:
            return self.buffer[:self.buffer_index].copy()

