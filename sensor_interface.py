from abc import ABC, abstractmethod
class SensorInterface(ABC):
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def extract(self):
        pass
