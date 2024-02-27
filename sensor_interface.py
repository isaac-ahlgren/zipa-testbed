import multiprocessing as mp

class SensorInterface():
    def __init__(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.poll)


    # TODO start, stop, read, must be implemented on a sensor basis
    def start(self):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
    
    def read(self, chunk_size):
        raise NotImplementedError
    
    def extract(self):
        return self.queue.get()
    
    def poll(self):
        while True:
            buf = self.read(self.chunk_size)
            self.queue.put(buf)

    def start_thread(self):
        self.process.start()