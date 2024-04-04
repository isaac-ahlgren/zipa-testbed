import multiprocessing as mp


import numpy as np

from nfs import NFSLogger


class SensorCollector:
    """
    Class copies `Sensor_Reader`, but refitted to collect and log sensor data
    onto NFS server and MySQL. Noticable difference is the absence of `read`.
    """

    def __init__(self, device, logger):
        self.sensor = device
        self.logger = logger
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=device.name)
        self.poll_process.start()

    def poll(self):
        while True:
            data = self.sensor.extract()
            csv = ", ".join(str(num) for num in data)
            self.logger.log((str(self.sensor.name), "csv", csv))

if __name__ == "__main__":
    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()


    from test_sensor import Test_Sensor

    ts = Test_Sensor(48000, 3 * 48000, 12_000, signal_type="random")
    sen_reader = SensorCollector(ts, NFSLogger(
                      user='luke',
                      password='lucor011&',
                      host='10.17.29.18',
                      database='file_log',
                      nfs_server_dir='/mnt/data',  # Make sure this directory exists and is writable
                      identifier='192.168.1.220'  # Could be IP address or any unique identifier
                    ))
