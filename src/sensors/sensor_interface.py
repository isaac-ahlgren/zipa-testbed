import multiprocessing as mp
from typing import Any

class SensorInterface:
    def __init__(self) -> None:
        pass

    # start, stop, read, must be implemented on a sensor basis
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def read(self) -> Any:
        raise NotImplementedError
