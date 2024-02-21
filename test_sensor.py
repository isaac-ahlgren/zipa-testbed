import numpy as np

class Test_Sensor:
    def __init__(self, sample_rate, buffer_size, signal_type='random'):
        self.sampling = sample_rate
        self.buffer_size = buffer_size
        
        self.time = 0
        self.buffer_ready = False
        self.ready_buffer = None
        self.buffer = None
        self.data_type = np.float32
        self.data_type_size = 4

        self.signal_type = signal_type

    def start(self):
        pass

    def stop(self):
        pass

    def extract(self):
        output = np.zeros(self.buffer_size, dtype=self.data_type)
        if self.signal_type == 'random':
            rng = np.random.default_rng()
            for i in range(len(output)):
                output[i] = rng.random()
        elif self.signal_type == 'sine':
            for i in range(len(output)):
                output[i] = np.sin(2*np.pi/self.sampling*i)
            self.time += len(output)
        return output
        
