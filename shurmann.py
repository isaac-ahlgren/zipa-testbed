import time

import mysql.connector
import numpy as np

from reed_solomon import ReedSolomonObj
from corrector import Fuzzy_Commitment
from network import *


# TODO: Make it so that key length is a parameter to the Shurmann and siggs protocol
# It will need work with how many bits the quantizer
# TODO: Make template for protocols so there is are guaranteed boiler plate functionality in how to initialize it
class Shurmann_Siggs_Protocol:
    def __init__(self, microphone, n, k, timeout, nfs_server_dir, identifier):
        self.signal_measurement = microphone
        self.re = Fuzzy_Commitment(ReedSolomonObj(n ,k), 8)
        self.name = "shurmann-siggs"
        self.count = 0
        self.timeout = timeout
        self.nfs_server_dir = nfs_server_dir
        self.identifier = identifier

    def sigs_algo(self, x1, window_len=10000, bands=1000):
        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
        
        FFTs = []
        from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x)

        x = np.array(x1.copy())
        # wind = scipy.signal.windows.hann(window_len)
        for i in range(0, len(x), window_len):
            if len(x[i : i + window_len]) < window_len:
                # wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i : i + window_len] = x[i : i + window_len]  # * wind
            else:
                x[i : i + window_len] = x[i : i + window_len]  # * wind

            FFTs.append(abs(rfft(x[i : i + window_len])))

        E = {}
        bands_lst = []
        for i in range(0, len(FFTs)):
            frame = FFTs[i]
            bands_lst.append(
                [frame[k : k + bands] for k in range(0, len(frame), bands)]
            )
            for j in range(0, len(bands_lst[i])):
                E[(i, j)] = np.sum(bands_lst[i][j])

        bs = ""
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if E[(i, j)] - E[(i, j + 1)] - (E[(i - 1, j)] - E[(i - 1, j + 1)]) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)

    # TODO: Add timestamp information
    def extract_context(self):
        print("\nExtracting Context\n")
        # Time audio gathering
        start = time.time()
        signal = self.signal_measurement.get_audio()
        stop = time.time()
        bits = self.sigs_algo(signal)

        return bits, signal, start, stop

    def device_protocol(self, host):
        host.setblocking(1)
        print(f"Iteration {str(self.count)}.\n")

        # Sending ack that they are ready to begin
        print("Sending ACK.\n")
        ack(host)

        # Wait for ack from host to being context extract, quit early if no response within time
        print("Waiting for ACK from host.")
        if not ack_standby(host, self.timeout):
            print("No ACK recieved within time limit - early exit.\n\n")
            return

        # TODO: Update this with time info
        # Extract bits from mic
        print("Extracting context\n")
        witness, signal, start, stop = self.extract_context()

        # Wait for Commitment
        print("Waiting for commitment from host")
        host_commitment, host_metadata = commit_standby(host, self.timeout)

        # Early exist if no commitment recieved in time
        if not host_commitment:
            print("No commitment recieved within time limit - early exit\n")
            return

        print(
            f"Witness:\n{str(hex(int(witness, 2)))}\nRecieved hash: {str(host_metadata['hash'])}\n"
        )

        C = None
        success = False
        sampling = self.signal_measurement.sample_rate
        client_time = stop - start
        APPROX = 500
        shifted_signal = np.array(signal.copy())

        # TODO: Align timing as close as possible, then begin to iterate through the thousands

        # shifted_time = (stop - host_metadata["time_start"]) if host_metadata["time_start"] > start else (host_metadata["time_stop"] - start)

        # If the host begins after the client
        if host_metadata["time_start"] > start:
            shifted_time = stop - host_metadata["time_start"]
            shifted_total_frames = int(shifted_time * sampling)
            shifted_signal = shifted_signal[-(shifted_total_frames - APPROX) :]
            print(
                f"Losing beginning frames. Total no. of frames: {shifted_total_frames}"
            )
        # Else if the host begins before the client
        elif host_metadata["time_start"] < start:
            shifted_time = host_metadata["time_stop"] - start
            shifted_total_frames = int(shifted_time * sampling)
            shifted_signal = shifted_signal[: shifted_total_frames + APPROX]
            print(f"Losing end frames. Total no. of frames: {shifted_total_frames}")

        for window in range(1000):
            # Debugging
            print(f"Iteration #{window}. Length: {len(shifted_signal)}\nC: {str(C)}\n Value: {shifted_signal[window]}")
            witness = self.sigs_algo(shifted_signal[window:])
            C, success = self.re.decommit_witness(
                host_commitment, witness, host_metadata["hash"]
            )
            if success:
                print(f"Success!\nC: {str(C)}")
                break

        print(f"Witness:\n{str(hex(int(witness, 2)))}\nRecieved hash: {str(host_metadata['hash'])}\n")

        C = None
        success = False
        sampling = self.microphone.sample_rate
        client_time = stop - start
        APPROX = 500
        shifted_signal = np.array(signal.copy())

        # TODO: Align timing as close as possible, then begin to iterate through the thousands

        # If the host begins after the client
        if host_metadata["time_start"] > start:
            shifted_time = stop - host_metadata["time_start"]
            shifted_total_frames = int(shifted_time * sampling)
            shifted_signal = shifted_signal[-(shifted_total_frames - APPROX) :]
        # Else if the host begins before the client
        elif host_metadata["time_start"] < start:
            shifted_time = host_metadata["time_stop"] - start
            shifted_total_frames = int(shifted_time * sampling)
            shifted_signal = shifted_signal[: shifted_total_frames + APPROX]

        for window in range(1000):
            # Debugging
            print(f"Shifting window. Iteration #{window}.\n")
            witness = self.sigs_algo(shifted_signal[window:])
            C, success = self.re.decommit_witness(
                host_commitment, witness, host_metadata["hash"]
            )
            if success:
                print("Success!\n")
                break

        # Decommit
        # print("Decommiting")
        # C, success = self.re.decommit_witness(commitment, witness, h)

        print(f"C: {str(C)}\nSuccess? {str(success)}\n")

        # Log all information to NFS server
        print("Logging all information to NFS server")
        self.send_to_nfs_server(
            "audio", signal, witness, host_metadata["hash"], host_commitment
        )

        self.count += 1

    def host_protocol(self, device_sockets):
        print("Iteration " + str(self.count))
        print()

        participating_sockets = ack_all_standby(device_sockets, self.timeout)

        # Exit early if no devices to pair with
        if len(participating_sockets) == 0:
            print("No advertised devices joined the protocol - early exit")
            print()
            return
        print("Successfully ACKed participating devices")
        print()

        print("ACKing all participating devices")
        ack_all(participating_sockets)

        # TODO: Update with time info
        # Extract key from mic
        print("Extracting Context")
        witness, signal, start, stop = self.extract_context()
        print()

        # Commit Secret
        print("Commiting Witness")
        secret_key, h, commitment = self.re.commit_witness(witness)

        print("witness: " + str(hex(int(witness, 2))))
        print("h: " + str(h))
        print()

        print("Sending commitment")
        print()
        send_commit(commitment, h, participating_sockets)

        # Log all information to NFS server
        print("Logging all information to NFS server")
        self.send_to_nfs_server("audio", signal, witness, h, commitment)

        self.count += 1

    # TODO: Have timestamp info sent to NFS
    def send_to_nfs_server(
        self, signal_type, signal, timestamp, witness, h, commitment
    ):
        root_file_name = self.nfs_server_dir + "/" + signal_type

        signal_file_name = (
            root_file_name
            + "_signal_id"
            + str(self.identifier)
            + "_it"
            + str(self.count)
            + ".csv"
        )
        witness_file_name = (
            root_file_name
            + "_witness_id"
            + str(self.identifier)
            + "_it"
            + str(self.count)
            + ".txt"
        )
        hash_file_name = (
            root_file_name
            + "_hash_id"
            + str(self.identifier)
            + "_it"
            + str(self.count)
            + ".txt"
        )
        commitment_file_name = (
            root_file_name
            + "_commitment_id"
            + str(self.identifier)
            + "_it"
            + str(self.count)
            + ".csv"
        )

        np.savetxt(signal_file_name, signal)
        np.savetxt(commitment_file_name, np.array(commitment.coeffs))
        with open(witness_file_name, "w") as text_file:
            text_file.write(witness)
        with open(hash_file_name, "w") as text_file:
            text_file.write(str(h))

        conn = None
        try:
            conn = mysql.connector.connect(
                user="luke",
                password="lucor011&",
                host="10.17.29.18",
                database="file_log",
            )
            cursor = conn.cursor()

            for file_name in [
                signal_file_name,
                witness_file_name,
                hash_file_name,
                commitment_file_name,
            ]:
                cursor.execute(
                    "INSERT INTO file_paths (file_path) VALUES (%s)", (file_name,)
                )

            conn.commit()
        except mysql.connector.Error as err:
            print("Error connecting to MySQL:", err)
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
