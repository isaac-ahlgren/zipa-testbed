import multiprocessing as mp
import time

import numpy as np
import serial
import serial.threaded

from sensors.sensor_interface import SensorInterface

SETUP = b"s"
ACCEPT = b"a"
CANCEL = b"x"
GO = b"g"
HALT = b"h"
RESET = b"r"
HELLO = b"y"
CONFIRM = b"Hello\r\n"
EOL = b"\r\n"
BOOT = b"Booting...\r\n"


class Voltkey(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size, verbose=False):
        SensorInterface.__init__(self)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.chunks = self.buffer_size // self.chunk_size
        self.name = "voltkey"
        self.buffer = np.zeros(chunk_size, np.float32())
        self.verbose = verbose
        self.data_type = self.buffer.dtype
        self.sensor = serial.Serial(port="/dev/ttyUSB0", baudrate=115200)  # or ttyUSB0
        self.started = mp.Value("i", 0)
        self.mutex = mp.Semaphore()
        self.start_thread()

    def start(self):
        with self.mutex:
            serial_message = b""

            if self.verbose:
                print(f"Resetting {self.name}.\n")
            self.sensor.write(RESET)

            while BOOT not in serial_message:
                serial_message = self.sensor.read_until(EOL)
                if self.verbose:
                    print(f"Recieved from sensor: {serial_message}")

            if self.verbose:
                print(f"Starting up {self.name} sensor.\n")
                print("Handshaking with sensor.\n")
                print(f"Sensor name: {self.sensor.name}")
            self.sensor.write(HELLO)

            if self.verbose:
                print("Waiting for sensor to acknowledge.\n")
            serial_message = self.sensor.read_until(EOL)

            if serial_message != CONFIRM:
                if self.verbose:
                    print("Handshake failed. Aborting.\n")

                return

            if self.verbose:
                print(
                    f"Sending chunk size of argument. Chunk size: {self.chunk_size}.\n"
                )
            self.sensor.write(SETUP)

            if self.verbose:
                serial_message = self.sensor.read_until(EOL)
                print(f"Recieved from sensor: {serial_message.decode()}.\n")

            for char in str(self.chunk_size):
                self.sensor.write(char.encode())

            self.sensor.write(ACCEPT)

            if self.verbose:
                serial_message = self.sensor.read_until(EOL)
                print(f"Recieved from sensor: {serial_message.decode()}.\n")
                serial_message = self.sensor.read_until(EOL)
                print(f"Recieved from sensor: {serial_message.decode()}.\n")

            self.sensor.write(GO)

            serial_message = self.sensor.read_until(EOL)
            self.started.value = 1

    def stop(self):
        serial_message = b""

        with self.mutex:
            if self.verbose:
                print(f"Stopping {self.sensor.name} reading.\n")

            self.sensor.write(RESET)

            while BOOT not in serial_message:
                serial_message = self.sensor.read_until(EOL)
                if self.verbose:
                    print(f"Recieved from sensor: {serial_message}")

    def read(self):
        # Read doesn't get mutex lock until sensor's started
        if self.started.value == 0:
            time.sleep(0.1)

        with self.mutex:
            signal = self.sensor.read_until(EOL)
            raw_signal = str(signal).split("\\x")
            raw_signal = [num for num in raw_signal if num.isdigit()]
            cleaned_signal = []

            # Removes bytestream prefix, escape character suffxies; expecting short int values
            for hexadecimal in range(1, len(raw_signal) - 1, 2):
                cleaned_signal.append(
                    int(raw_signal[hexadecimal] + raw_signal[hexadecimal + 1], 16)
                )

            # print(cleaned_signal)
            data = np.array(cleaned_signal, dtype=self.data_type)

            return data


if __name__ == "__main__":
    from sensors.sensor_reader import Sensor_Reader

    sample_rate = 2000
    buffer_size = sample_rate * 17
    chunk_size = sample_rate * 4

    voltkey = Voltkey(sample_rate, buffer_size, chunk_size)
    sr = Sensor_Reader(voltkey)
    time.sleep(3)

    print("Beginning reading.")

    for i in range(10):
        # sr.sensor.sensor.write(GO)
        results = sr.read(sample_rate * 17)
        print(f"Number of results: {len(results)},\n{results}")
        # time.sleep(10)

    exit()
