import multiprocessing as mp

from networking.nfs import NFSLogger


class Sensor_Collector:
    """
    A class designed to manage sensor data collection in a multiprocessing environment, logging the
    collected data using NFSLogger.

    :param device: The sensor device from which data is to be collected.
    :param logger: An instance of NFSLogger to log the collected sensor data.
    """
    def __init__(self, device, logger):
        """
        Initializes the Sensor_Collector with a specified sensor device and logger.

        :param device: The sensor device from which data is to be collected.
        :param logger: An NFSLogger to handle logging the collected data.
        """
        self.sensor = device
        self.logger = logger
        self.sensor.start()
        self.poll_process = mp.Process(target=self.poll, name=device.name)
        self.poll_process.start()

    def poll(self):
        """
        Continuously polls data from the sensor and logs it using the configured NFSLogger.
        This function runs in a separate process.
        """
        while True:
            data = self.sensor.read()
            self.logger.log_signal(self.sensor.name, data)


if __name__ == "__main__":

    def sen_thread(sen):
        p = mp.Process(target=sen.poll)
        p.start()

    from sensors.test_sensor import TestSensor

    ts = TestSensor(48000, 3 * 48000, 12_000, signal_type="random")
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
