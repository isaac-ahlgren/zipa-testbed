import numpy as np
import serial, serial.threaded
import time
from sensor_interface import SensorInterface

SETUP   = 's'
ACCEPT  = 'a'
CANCEL  = 'x'
GO      = 'g'
HALT    = 'h'
RESET   = 'r'
HELLO   = 'y'
CONFIRM = 'Hello'


class Voltkey(SensorInterface):
    def __init__(self, sample_rate, buffer_size, chunk_size, verbose=True):
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
        self.timeout = 10
        self.sensor = serial.Serial(
            port="/dev/ttyUSB0", # or ttyUSB0
            baudrate=115200
        )
        self.start_thread()

    def start(self):
        # serial_message = b''
        # self.sensor.flush()
        # self.sensor.write(RESET)

        """
        while serial_message != b'Booting...\r\n':
            print("In reset loop.")
            self.sensor.write(RESET)
            serial_message = self.sensor.read_until(EOL)
            print(serial_message)
            serial_message = self.sensor.read_until(EOL)
            print(serial_message)
        """
        # serial_message = self.sensor.read_until(EOL)
        # print(serial_message)
        # serial_message = self.sensor.read_until(EOL)
        # print(serial_message)

        with serial.threaded.ReaderThread(self.sensor, Reciever) as protocol:
            if self.verbose:
                print(f"Starting up {self.name} sensor.\n Handshaking with sensor.\n")
            protocol.write_line(HELLO)

            while not protocol.recieved_packets: pass
            serial_msg = protocol.recieved_packets.pop(0).decode()
            if self.verbose:
                print(f"Recieved packets: {serial_msg}.\n")

            if serial_msg != CONFIRM:
                if self.verbose:
                    print("Handshake failed. Aborting.\n")

                return
            
            if self.verbose:
                print(f"Sending chunk size of argument. Chunk size: {self.chunk_size}.\n")
            protocol.write_line(SETUP + str(self.chunk_size) + ACCEPT)
            
            while not protocol.recieved_packets: pass
            serial_msg = protocol.recieved_packets.pop(0).decode()
            if self.verbose:
                print(f"Recieved packets: {serial_msg}")

            while not protocol.recieved_packets: pass
            serial_msg = protocol.recieved_packets.pop(0).decode()
            
            if self.verbose:
                print(f"Recieved packets: {serial_msg}")

            while not protocol.recieved_packets: pass
            serial_msg = protocol.recieved_packets.pop(0).decode()
            
            if self.verbose:
                print(f"Recieved packets: {serial_msg}")

            protocol.write_line(GO)
            while not protocol.recieved_packets: pass
            serial_msg = protocol.recieved_packets.pop(0).decode()

            if self.verbose:
                print(f"Recieved packets: {serial_msg}")


    def stop(self):
        if self.verbose:
            print(f"Stopping {self.sensor.name} reading.\n")

        self.sensor.write(RESET)

        if self.verbose:
            serial_message = self.sensor.read_until(EOL)
            print(f"Recieved from sensor: {serial_message.decode()}.\n")

    def read(self):
        time.sleep(10)
        print("In voltkey read function.")
        # self.sensor.open()
        # signal = self.interface.run()
        # self.sensor.close()
        # raw_signal = str(signal).split('\\x')
        raw_signal = [num for num in raw_signal if num.isdigit()]
        cleaned_signal = []

        # Removes bytestream prefix, escape character suffxies; expecting short int values
        for hexadecimal in range(1, len(raw_signal) - 1, 2):
            cleaned_signal.append(int(raw_signal[hexadecimal] + raw_signal[hexadecimal + 1], 16))

        # print(cleaned_signal)
        data = np.array(cleaned_signal, dtype=self.data_type)

        return data
    
    """
    def poll(self):
        if self.started:
            self.start()
            self.started = True

        while True:
            buf = self.read()
            self.queue.put(buf)
    """

class Reciever(serial.threaded.LineReader):
    TERMINATOR = b'\r\n'

    def __init__(self):
        super().__init__()
        self.recieved_packets = []
    
    def handle_packet(self, packet):
        self.recieved_packets.append(packet)

    def handle_line(self, line) -> None:
        self.recieved_packets.append(line)

if __name__ == "__main__":
    from sensor_reader import Sensor_Reader
    import time

    timeout = time.time() + 10

    sample_rate = 25
    buffer_size = sample_rate * 17
    chunk_size = sample_rate * 4

    voltkey = Voltkey(sample_rate, buffer_size, chunk_size)
    sr = Sensor_Reader(voltkey)
    time.sleep(3)
    
    print("Beginning reading.")
    
    for i in range(10):
        # sr.sensor.sensor.write(GO)
        results = sr.read(sample_rate * 17) # TODO populate arguments
        print(f"Number of results: {len(results)},\n{results}")
        time.sleep(10)

    exit()

