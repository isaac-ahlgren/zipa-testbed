import os
from datetime import datetime

import mysql.connector
import numpy as np
import pandas as pd


class NFSLogger:
    def __init__(
        self,
        user,
        password,
        host,
        database,
        nfs_server_dir,
        local_dir,
        identifier,
        use_local_dir=False,
    ):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier
        self.local_dir = local_dir
        self.use_local_dir = use_local_dir

    def log_signal(self, name, signal):
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
        file_name = (
            directory  # On NFS or locally
            + name  # Sensor name
            + "_id_"
            + str(self.identifier)  # Unique ID for deivce
            + "_date_"
            + timestamp  # Day
            + ".csv"
        )

        # Convert to CSV and save
        df = pd.DataFrame(signal)
        df.to_csv(file_name, mode="a", header=False, index=False)

    def log(self, data_tuples, count=None, ip_addr=None):
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

    def create_filename(self, directory, name, count, ip_addr, file_ext):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{directory}/{name}_id{self.identifier}_{timestamp}"
        if count is not None:
            filename += f"_count{count}"
        if ip_addr is not None:
            filename += f"_toipaddr{ip_addr}"
        filename += f".{file_ext}"

        return filename

    # TODO: Check if mySQL will be used for project
    def _log_to_mysql(self, file_paths):
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
