import numpy as np
import serial

from sensor_interface import SensorInterface

SETUP = "s".encode()
ACCEPT = "a".encode()
CANCEL = "x".encode()
GO = "g".encode()
HALT = "h".encode()
RESET = "r".encode()
HELLO = "y".encode()
EOL = "\r\n".encode()


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
        self.buffer_index = 0
        self.data_type = self.buffer.dtype
        self.sensor = serial.Serial("/dev/ttyS0")
        self.sensor.baudrate = 9600  # TODO Verify if a different rate is needed
        self.start_thread()

    def start(self):
        if self.verbose:
            print(f"Starting up {self.name} sensor.\n")
            print("Handshaking with sensor.\n")
        self.sensor.write(HELLO)

        if self.verbose:
            print("Waiting for sensor to acknowledge.\n")
        serial_message = self.sensor.read_until(EOL)

        if serial_message != HELLO:
            if self.verbose:
                print("Handshake failed. Aborting.\n")

            return

        if self.verbose:
            print(f"Sending chunk size of argument. Chunk size: {self.chunk_size}.\n")
        self.sensor.write(SETUP)

        if self.verbose:
            serial_message = self.sensor.read_until(EOL)
            print(f"Recieved from sensor: {serial_message.decode()}.\n")

        for char in str(self.chunk_size):
            self.sensor.write(char.encode())

            if self.verbose:
                serial_message = self.sensor.read_until(EOL)
                print(f"Recieved from sensor: {serial_message.decode()}.\n")

        self.sensor.write(ACCEPT)

        if self.verbose:
            serial_message = self.sensor.read_until(EOL)
            print(f"Recieved from sensor: {serial_message.decode()}.\n")

    def stop(self):
        if self.verbose:
            print(f"Stopping {self.sensor.name} reading.\n")

        self.sensor.write(CANCEL)

        if self.verbose:
            serial_message = self.sensor.read_until(EOL)
            print(f"Recieved from sensor: {serial_message.decode()}.\n")

    def read(self):
        signal = self.sensor.read(self.chunk_size)
        data = np.array(signal, dtype=self.data_type)

        return data


"""
if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time

    voltkey = Voltkey() # TODO populate arguments
    sr = Sensor_Reader(voltkey)
    time.sleep(3)
    print("Beginning reading.")

    for i in range(10):
        results = sr.read() # TODO populate arguments
        print(f"Number of results: {len(results)},\n{results}")
        time.sleep(10)

    exit()
"""
