import numpy as np

class Signal_File_Interface:

    def read(self, samples: int) -> np.ndarray:
        pass

    def get_finished_reading(self):
        pass

    def get_id(self):
        pass

    def reset(self):
        pass

    def sync(self, other_sf):
        pass