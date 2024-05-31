import multiprocessing as mp

from nfs import NFSLogger


class Sensor_Collector:

    def __init__(self, device, logger):
        self.sensor = device
        self.logger = logger
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=device.name)
        self.poll_process.start()

    def poll(self):
        while True:
            data = self.sensor.extract()
            self.logger.log_signal(self.sensor.name, data)


if __name__ == "__main__":

    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()

    from test_sensor import Test_Sensor

    ts = Test_Sensor(48000, 3 * 48000, 12_000, signal_type="random")
    sen_reader = Sensor_Collector(
        ts,
        NFSLogger(
            user="USERNAME",
            password="PASSWORD",
            host="SERVER IP",
            database="file_log",
            nfs_server_dir="/mnt/data",  # Make sure this directory exists and is writable
            identifier="DEVICE ID",  # Could be IP address or any unique identifier
        ),
    )
