import multiprocessing
import RPi.GPIO as GPIO
import time
from sensor_interface import SensorInterface
import numpy as np

class PIRSensor(SensorInterface):
    def __init__(self, sample_rate, buffer_size, pin=12):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.pin = pin
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=self._monitor_sensor, args=(self.child_conn,))
        self.buffer = np.zeros(buffer_size, dtype=int)  # Store states as 1s and 0s
        self.buffer_index = 0
        self.buffer_full = False
        self.data_type = self.buffer.dtype

    def _monitor_sensor(self, conn):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)

        try:
            while True:
                current_state = GPIO.input(self.pin)
                # Send the current state (either 1 or 0)
                conn.send(current_state)

                if conn.poll():  # Check for the stop signal
                    if conn.recv() == 'STOP':
                        break

                time.sleep(1 / self.sample_rate)
        finally:
            GPIO.cleanup(self.pin)

    def start(self):
        self.process.start()

    def stop(self):
        self.parent_conn.send('STOP')
        self.process.join()

    def extract(self):
        while self.parent_conn.poll():
            state = self.parent_conn.recv()
            # Store the state directly into the buffer
            self.buffer[self.buffer_index] = state
            self.buffer_index += 1
            if self.buffer_index == self.buffer_size:
                self.buffer_index = 0
                self.buffer_full = True

        if self.buffer_full:
            return self.buffer.copy()
        else:
            return self.buffer[:self.buffer_index].copy()

