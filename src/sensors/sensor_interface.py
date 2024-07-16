from typing import Any


class SensorInterface:
    """
    An abstract base class for sensors that outlines the mandatory methods that need to be implemented
    by any specific sensor class inheriting from it. This ensures a consistent interface for starting,
    stopping, and reading data from various sensors.

    This class is meant to be inherited by sensor-specific classes that define actual hardware interaction
    logic tailored to their specific requirements.
    """

    def __init__(self) -> None:
        """
        Initializes the SensorInterface. This constructor is typically called by constructors of
        derived classes to ensure the base class is properly initialized, even though this base class
        itself does not hold any initialization logic.
        """
        pass

    # start, stop, read, must be implemented on a sensor basis
    def start(self) -> None:
        """
        Starts the sensor operation. This method needs to be overridden by derived classes to include
        logic to initiate the sensor's data collection.

        :raises NotImplementedError: If the method is not implemented by the derived class.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """
        Stops the sensor operation. This method needs to be overridden by derived classes to include
        logic to stop the sensor's data collection safely and cleanly.

        :raises NotImplementedError: If the method is not implemented by the derived class.
        """
        raise NotImplementedError

    def read(self) -> Any:
        """
        Reads data from the sensor. This method must be implemented by derived classes to handle the
        specifics of data acquisition from the sensor.

        :raises NotImplementedError: If the method is not implemented by the derived class.
        """
        raise NotImplementedError
