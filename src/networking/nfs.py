import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import mysql.connector
import numpy as np
import pandas as pd


class NFSLogger:
    """
    A logger class designed to log data locally or on a network file system (NFS) server,
    and optionally record file paths to a MySQL database. It supports logging different types
    of data including floats, bytes, and CSV data.

    :param user: MySQL database user name.
    :param password: MySQL database password.
    :param host: Hostname of the MySQL server.
    :param database: Name of the MySQL database.
    :param nfs_server_dir: Directory path on the NFS where files will be stored.
    :param local_dir: Local directory path where files will be stored if not using NFS.
    :param identifier: A unique identifier for the logging instance.
    :param use_local_dir: A boolean indicating whether to use the local directory for storage instead of NFS.
    """

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        database: str,
        nfs_server_dir: str,
        local_dir: str,
        identifier: str,
        use_local_dir: bool = False,
    ) -> None:
        """
        Initializes the NFSLogger with database and directory information.
        """
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier
        self.local_dir = local_dir
        if not os.path.isdir(local_dir):
            os.mkdir(local_dir)
        self.use_local_dir = use_local_dir
        if not os.path.isdir(self.local_dir):
            os.mkdir(self.local_dir)

    def log_signal(self, name: str, signal: Union[float, List[float]]) -> None:
        """
        Logs sensor signal data to a file. The data can be a single float or a list of floats.

        :param name: The name of the sensor or data source.
        :param signal: The signal data to log, can be a float or a list of floats.
        """
        """
        Used in the sensor collector configuration.

        """
        if isinstance(signal, float):
            signal = [signal]

        # Dependant on selection in main.py
        if self.use_local_dir:
            directory = self.local_dir
        else:
            directory = self.nfs_server_dir

        # Logging one day's worth of signals into a file
        timestamp = datetime.now().strftime("%Y%m%d%H")
        file_name = f"{directory}{name}_id_{str(self.identifier)}_date_{timestamp}.csv"

        # Convert to CSV and save
        df = pd.DataFrame(signal)
        df.to_csv(file_name, mode="a", header=False, index=False)

    def log(
        self,
        data_tuples: List[Tuple[str, str, Union[str, bytes, List[bytes]]]],
        count: Optional[int] = None,
        ip_addr: Optional[str] = None,
    ) -> None:
        """
        Logs arbitrary data from various sources as specified in `data_tuples`. Data can be strings, bytes, or CSV-formatted.

        :param data_tuples: A list of tuples containing (name, file_ext, data).
        :param count: An optional count of data entries, used in file naming.
        :param ip_addr: An optional IP address, used in file naming.
        """
        """
        Used in the ZIPA protocol configuration.

        """

        # Dependant on selection in main.py
        if self.use_local_dir:
            directory = self.local_dir
        else:
            directory = self.nfs_server_dir

        for name, file_ext, data in data_tuples:
            # Create filename
            filename = self.create_filename(directory, name, count, ip_addr, file_ext)

            # Determine mode based on whether data is bytes or str (it can be a list of bytes too)
            if isinstance(data, bytes) or (
                isinstance(data, list) and isinstance(data, bytes)
            ):
                mode = "wb"
            else:
                mode = "w"

            # Open File and perform operation
            with open(filename, mode) as file:
                # If the file is CSV and data is not a string, use numpy to save
                if file_ext == "csv" and not isinstance(data, str):
                    np.savetxt(file, data, comments="", fmt="%s")
                elif mode == "wb":
                    if isinstance(data, list) and isinstance(data, bytes):
                        for byte_string in data:
                            file.write(byte_string)
                    else:
                        file.write(data)  # Data should already be bytes
                else:
                    file.write(
                        str(data)
                    )  # Data is a string, no encoding needed in text mode

            # log file path to mysql database
            if not self.local_dir:
                self._log_to_mysql([filename])

    def create_filename(
        self,
        directory: str,
        name: str,
        count: Optional[int],
        ip_addr: Optional[str],
        file_ext: str,
    ) -> str:
        """
        Creates a filename with a timestamp, name, count, and IP address.

        :param directory: The base directory for file storage.
        :param name: The name of the data source.
        :param count: An optional count of data entries, used in file naming.
        :param ip_addr: An optional IP address, used in file naming.
        :param file_ext: The file extension to use.
        :returns: A fully qualified file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{directory}/{name}_id{self.identifier}_{timestamp}"
        if count is not None:
            filename += f"_count{count}"
        if ip_addr is not None:
            filename += f"_toipaddr{ip_addr}"
        filename += f".{file_ext}"

        return filename

    # TODO: Check if mySQL will be used for project
    def _log_to_mysql(self, file_paths: List[str]) -> None:
        """
        Logs file paths to a MySQL database.

        :param file_paths: A list of file paths to log.
        """
        try:
            conn = mysql.connector.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.database,
            )
            cursor = conn.cursor()
            for file_path in file_paths:
                cursor.execute(
                    "INSERT INTO file_paths (file_path) VALUES (%s)", (file_path,)
                )
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
