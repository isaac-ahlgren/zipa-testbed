import struct

import numpy as np
from cryptography.hazmat.primitives import constant_time
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sklearn.cluster import KMeans

from networking.network import *
from protocols.common_protocols import *
from protocols.protocol_interface import ProtocolInterface


class Perceptio_Protocol(ProtocolInterface):
    def __init__(self, parameters, sensor, logger):
        ProtocolInterface.__init__(self, parameters, sensor, logger)
        self.name = "FastZIP_Protocol"
        self.wip = True
        self.a = parameters["a"]
        self.cluster_sizes_to_check = parameters["cluster_sizes_to_check"]
        self.cluster_th = parameters["cluster_th"]
        self.top_th = parameters["top_th"]
        self.bottom_th = parameters["bottom_th"]
        self.lump_th = parameters["lump_th"]
        self.conf_threshold = parameters["conf_thresh"]
        self.max_iterations = parameters["max_iterations"]
        self.sleep_time = parameters["sleep_time"]
        self.max_no_events_detected = parameters["max_no_events_detected"]
        self.time_length = parameters["time_length"]
        self.nonce_byte_size = 16
        self.count = 0

    def extract_context(self, socket):
        events_detected = False
        for i in range(self.max_no_events_detected):
            signal = self.sensor.read(self.time_length)
            fps, events = self.perceptio(
                signal,
                self.commitment_length,
                self.sensor.sensor.sample_rate,
                self.a,
                self.cluster_sizes_to_check,
                self.cluster_th,
                self.bottom_th,
                self.top_th,
                self.lump_th,
            )

            # Check if fingerprints were generated
            if len(fps) > 0:
                events_detected = True

            # Send current status
            send_status(socket, events_detected)

            # Check if other device also succeeded
            status = status_standby(socket, self.timeout)

            if status == None:
                events_detected = status
            else:
                events_detected = events_detected and status

            # Break out of the loop if event was detected
            if events_detected:
                break

            time.sleep(self.sleep_time)

        return fps, signal, events_detected

    #  TODO: Fix why this does not save correctly to drive
    def parameters(self, is_host):
        parameters = f"protocol: {self.name} is_host: {str(is_host)}\n"
        parameters += f"sensor: {self.sensor.sensor.name}\n"
        parameters += f"key_length: {self.key_length}\n"
        parameters += f"parity_symbols: {self.parity_symbols}\n"
        parameters += f"a: {self.a}\n"
        parameters += f"cluster_sizes_to_check: {self.cluster_sizes_to_check}\n"
        parameters += f"cluster_th: {self.cluster_th}\n"
        parameters += f"top_th: {self.cluster_th}\n"
        parameters += f"bottom_th: {self.bottom_th}\n"
        parameters += f"time_length: {self.time_length}\n"

    def device_protocol(self, host_socket):
        host_socket.setblocking(1)

        if self.verbose:
            print("Iteration " + str(self.count))

        # Log current paramters to the NFS server
        self.logger.log([("parameters", "txt", self.parameters(False))])

        # Sending ack that they are ready to begin
        if self.verbose:
            print("\nSending ACK")
        ack(host_socket)

        successes = 0
        iterations = 0
        while successes < self.conf_threshold and iterations < self.max_iterations:
            success = False

            if self.verbose:
                print("Waiting for ACK from host.\n")
            if not ack_standby(host_socket, self.timeout):
                if self.verbose:
                    print("No ACK recieved within time limit - early exit.\n\n")
                return

            if self.verbose:
                print("Extracting context\n")

            # Extract bits from sensor
            witnesses, signal, status = self.extract_context(host_socket)

            if status == None:
                if self.verbose:
                    print(
                        "Other device did not respond during extraction - early exit\n"
                    )
                return
            elif not status:
                if self.verbose:
                    print("Not enough events detected: moving to next iteration")
                continue

            if self.verbose:
                print("Waiting for commitment from host\n")
            commitments, hs = commit_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not commitments:
                if self.verbose:
                    print("No commitment recieved within time limit - early exit\n")
                return

            if self.verbose:
                print("Commitments recieved\n")
                print("Uncommiting with witnesses\n")
            key = self.find_commitment(commitments, hs, witnesses)

            success = key != None
            send_status(host_socket, success)

            # TODO: Fails to uncommit only sometimes which is weird
            # Commitment failed, try again
            if key == None:
                if self.verbose:
                    print(
                        "Witnesses failed to uncommit any commitment - alerting other device for a retry\n"
                    )
                self.checkpoint_log(witnesses, commitments, success, signal, iterations)
                iterations += 1
                continue

            # Key Confirmation Phase

            if self.verbose:
                print("Performing key confirmation\n")

            # Derive key
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(key)

            # Hash prederived key
            pd_key_hash = self.hash_function(key)

            # Send nonce message to host
            generated_nonce = send_nonce_msg_to_host(
                host_socket,
                pd_key_hash,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            )

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(host_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                return

            # If hashes are equal, then it was successful
            if verify_mac_from_host(
                recieved_nonce_msg,
                generated_nonce,
                derived_key,
                self.nonce_byte_size,
                self.hash_func,
            ):
                success = True
                successes += 1

            if self.verbose:
                print(f"Produced Key: {derived_key}\n")
                print(
                    f"success: {success}, Number of successes {successes}, Total number of iteration {iterations}\n"
                )

            self.checkpoint_log(witnesses, commitments, success, signal, iterations)
            iterations += 1

        if self.verbose:
            print(
                f"Total Key Pairing Result: success - {successes >= self.conf_threshold}\n"
            )

        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {successes} total_iterations: {iterations} succeeded: {successes >= self.conf_threshold})",
                )
            ]
        )

    def host_protocol_single_threaded(self, device_socket):
        device_ip_addr, device_port = device_socket.getpeername()

        # Exit early if no devices to pair with
        if not ack_standby(device_socket, self.timeout):
            if self.verbose:
                print("No ACK recieved within time limit - early exit.\n\n")
            return
        if self.verbose:
            print("Successfully ACKed participating device")
            print()

        successes = 0
        iterations = 0
        while successes < self.conf_threshold and iterations < self.max_iterations:
            success = False

            if self.verbose:
                print("ACKing all participating devices")
            ack(device_socket)

            if self.verbose:
                print("Extracting context\n")
            # Extract bits from sensor
            witnesses, signal, status = self.extract_context(device_socket)

            if status == None:
                if self.verbose:
                    print(
                        "Other device did not respond during extraction - early exit\n"
                    )
                return
            elif not status:
                if self.verbose:
                    print("Not enough events detected: moving to next iteration")
                continue

            if self.verbose:
                print("Commiting all the witnesses\n")
            # Create all commitments
            commitments, keys, hs = self.generate_commitments(witnesses)

            if self.verbose:
                print("Sending commitments\n")
            # Send all commitments

            print(witnesses)

            send_commit(commitments, hs, device_socket)

            # Check up on other devices status
            status = status_standby(device_socket, self.timeout)
            if status == None:
                if self.verbose:
                    print("No status recieved within time limit - early exit.\n\n")
                return
            elif status == False:
                if self.verbose:
                    print(
                        "Other device did not uncommit with witnesses - trying again\n"
                    )
                success = False
                self.checkpoint_log(
                    witnesses,
                    commitments,
                    success,
                    signal,
                    iterations,
                    ip_addr=device_ip_addr,
                )
                iterations += 1
                continue

            # Key Confirmation Phase

            # Recieve nonce message
            recieved_nonce_msg = get_nonce_msg_standby(device_socket, self.timeout)

            # Early exist if no commitment recieved in time
            if not recieved_nonce_msg:
                if self.verbose:
                    print("No nonce message recieved within time limit - early exit")
                    print()
                return

            derived_key = self.host_verify_mac(keys, recieved_nonce_msg)

            if derived_key:
                success = True
                successes += 1

                # Create and send key confirmation value
                send_nonce_msg_to_device(
                    device_socket,
                    recieved_nonce_msg,
                    derived_key,
                    hs[0],
                    self.nonce_byte_size,
                    self.hash_func,
                )

                self.checkpoint_log(
                    witnesses,
                    commitments,
                    success,
                    signal,
                    iterations,
                    ip_addr=device_ip_addr,
                )

            if self.verbose:
                print(
                    f"success: {success}, Number of successes {successes}, Total number of iterations {iterations}\n"
                )

            iterations += 1

        if self.verbose:
            print(
                f"Total Key Pairing Result: success - {successes >= self.conf_threshold}\n"
            )
        self.logger.log(
            [
                (
                    "pairing_statistics",
                    "txt",
                    f"successes: {successes} total_iterations: {iterations} succeeded: {successes >= self.conf_threshold})",
                )
            ],
            ip_addr=device_ip_addr,
        )

    def hash_function(self, bytes):
        hash_func = hashes.Hash(self.hash_func)
        hash_func.update(bytes)
        return hash_func.finalize()

    def ewma(signal, a):
        y = np.zeros(len(signal))

        y[0] = a * signal[0]
        for i in range(1, len(signal)):
            y[i] = a * signal[i] + (1 - a) * y[i - 1]
        return y

    def get_events(signal, a, bottom_th, top_th, lump_th):
        signal = self.ewma(np.abs(signal), a)

        # Get events that are within the threshold
        events = []
        found_event = False
        beg_event_num = None
        for i in range(len(signal)):
            if not found_event and signal[i] >= bottom_th and signal[i] <= top_th:
                found_event = True
                beg_event_num = i
            elif found_event and (signal[i] < bottom_th or signal[i] > top_th):
                found_event = False
                found_event = None
                events.append((beg_event_num, i))
        if found_event:
            events.append((beg_event_num, i))

        i = 0
        while i < len(events) - 1:
            if events[i + 1][0] - events[i][1] <= lump_th:
                new_element = (events[i][0], events[i + 1][1])
                events.pop(i)
                events.pop(i)
                events.insert(i, new_element)
            else:
                i += 1

        return events

    def get_event_features(events, signal):
        event_features = []
        for i in range(len(events)):
            length = events[i][1] - events[i][0]
            max_amplitude = np.max(signal[events[i][0] : events[i][1]])
            event_features.append((length, max_amplitude))
        return event_features

    def kmeans_w_elbow_method(event_features, cluster_sizes_to_check, cluster_th):
        if len(event_features) < cluster_sizes_to_check:
            # Handle the case where the number of samples is less than the desired number of clusters
            if self.verbose:
                print(
                    "Warning: Insufficient samples for clustering. Returning default label and k=1."
                )
            return np.zeros(len(event_features), dtype=int), 1

        km = KMeans(1, n_init=50, random_state=0).fit(event_features)
        x1 = km.inertia_
        rel_inert = x1

        k = None
        labels = None
        inertias = [rel_inert]

        for i in range(2, cluster_sizes_to_check):
            labels = km.labels_

            km = KMeans(i, n_init=50, random_state=0).fit(event_features)
            x2 = km.inertia_

            inertias.append(x2)
            perc = (x1 - x2) / rel_inert

            x1 = x2

            # Break if reached elbow
            if perc <= cluster_th:
                k = i - 1
                break

            # Break if reached end
            if i == cluster_sizes_to_check - 1:
                labels = km.labels_
                k = i
                break

        return labels, k

    def group_events(events, labels, k):
        event_groups = [[] for i in range(k)]
        for i in range(len(events)):
            event_groups[labels[i]].append(events[i])
        return event_groups

    def gen_fingerprints(grouped_events, k, key_size, Fs):
        from datetime import timedelta

        fp = []
        for i in range(k):
            event_list = grouped_events[i]
            key = bytearray()
            for j in range(len(event_list) - 1):
                interval = (event_list[j][0] - event_list[j + 1][0]) / Fs
                in_microseconds = int(
                    timedelta(seconds=interval) / timedelta(microseconds=1)
                )
                key += in_microseconds.to_bytes(
                    4, "big"
                )  # Going to treat every interval as a 4 byte integer

            if len(key) >= key_size:
                key = bytes(key[:key_size])
                fp.append(key)
        return fp

    def perceptio(
        signal,
        key_size,
        Fs,
        a,
        cluster_sizes_to_check,
        cluster_th,
        bottom_th,
        top_th,
        lump_th,
    ):
        events = Perceptio_Protocol.get_events(signal, a, bottom_th, top_th, lump_th)
        if len(events) < 2:
            # Needs two events in order to calculate interevent timings
            if self.verbose:
                print("Error: Less than two events detected")
            return ([], events)

        event_features = Perceptio_Protocol.get_event_features(events, signal)

        labels, k = Perceptio_Protocol.kmeans_w_elbow_method(
            event_features, cluster_sizes_to_check, cluster_th
        )

        grouped_events = Perceptio_Protocol.group_events(events, labels, k)

        fps = Perceptio_Protocol.gen_fingerprints(grouped_events, k, key_size, Fs)

        return fps, grouped_events

    def host_verify_mac(self, keys, recieved_nonce_msg):
        key_found = None
        for i in range(len(keys)):
            kdf = HKDF(
                algorithm=self.hash_func, length=self.key_length, salt=None, info=None
            )
            derived_key = kdf.derive(keys[i])
            key_hash = self.hash_function(keys[i])

            if verify_mac_from_device(
                recieved_nonce_msg,
                derived_key,
                key_hash,
                self.nonce_byte_size,
                self.hash_func,
            ):
                key_found = derived_key
                break
        return key_found

    def find_commitment(self, commitments, hashes, fingerprints):
        key = None
        for i in range(len(fingerprints)):
            for j in range(len(commitments)):
                potential_key = self.re.decommit_witness(
                    commitments[j], fingerprints[i]
                )
                potential_key_hash = self.hash_function(potential_key)
                if constant_time.bytes_eq(potential_key_hash, hashes[j]):
                    key = potential_key
                    break
        return key

    def generate_commitments(self, witnesses):
        commitments = []
        keys = []
        hs = []
        for i in range(len(witnesses)):
            key, commitment = self.re.commit_witness(witnesses[i])
            commitments.append(commitment)
            keys.append(key)
            hs.append(self.hash_function(key))
        return commitments, keys, hs

    def checkpoint_log(
        self, witnesses, commitments, success, signal, iterations, ip_addr=None
    ):
        self.logger.log(
            [
                ("witness", "txt", witnesses),
                ("commitments", "txt", commitments),
                ("success", "txt", str(success)),
                ("signal", "csv", signal),
            ],
            count=iterations,
            ip_addr=ip_addr,
        )


###TESTING CODE###
import socket


def device(prot):
    print("device")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(("127.0.0.1", 2000))
    prot.device_protocol(s)


def host(prot):
    print("host")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 2000))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(0)
    prot.host_protocol([conn])


# TODO: Update test case with new protocol arguments
if __name__ == "__main__":
    import multiprocessing as mp

    from networking.nfs import NFSLogger
    from sensors.sensor_reader import Sensor_Reader
    from sensors.test_sensor import TestSensor

    prot = Perceptio_Protocol(
        Sensor_Reader(TestSensor(44100, 44100 * 50, 1024, signal_type="random")),
        8,
        4,
        44100 * 20,
        0.3,
        3,
        0.08,
        0.75,
        0.5,
        5,
        5,
        20,
        5,
        10,
        10,
        NFSLogger(None, None, None, None, None, 1, "./data"),
    )
    h = mp.Process(target=host, args=[prot])
    d = mp.Process(target=device, args=[prot])
    h.start()
    d.start()
