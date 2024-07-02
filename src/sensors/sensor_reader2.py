import multiprocessing as mp
import select

import numpy as np

MAX_CLIENTS = 1024


class SensorReader:
    def __init__(self, sensor):
        self.sensor = sensor
        self.sensor_pipes = []
        self.protocol_pipes = []
        self.semaphore = mp.Semaphore(MAX_CLIENTS)
        self.mutex = mp.Semaphore()

    def poll(self):
        while True:
            # Holds a list of pipes to protocols
            pipes = None
            data = self.sensor.extract()

            with self.mutex:
                pipes, _, _ = select.select(pipes, [], [], [])

                # Send data to protocols if its requested
                for pipe in pipes:
                    send = pipe.recv()

                    if send:
                        pipe.send(data)

    def create_pipe(self):
        sensor_pipe, protocol_pipe = mp.pipe()

        # Keep track for listening and sending data, and closing pipes
        with self.semaphore:
            self.sensor_pipes.append(sensor_pipe)
            self.protocol_pipes.append(protocol_pipe)

        return protocol_pipe
    
    def close_pipe(self, protocoL_pipe):
        # Find protocol pipe index and close both protocol and sensor pipes
        pipe_index = self.protocol_pipes.index(protocoL_pipe)
        sensor_pipe = self.sensor_pipes[pipe_index]
        protocoL_pipe.close()
        sensor_pipe.close()

        # Free memory
        with self.semaphore:
            del self.sensor_pipes[pipe_index]
            del self.protocol_pipes[pipe_index]
