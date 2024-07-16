# import multiprocessing as mp


class SensorInterface:
    def __init__(self):
        pass

    # start, stop, read, must be implemented on a sensor basis
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError
