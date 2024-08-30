import numpy as np


class MiettinenProcessing:
    def signal_preprocessing(
        signal: np.ndarray, no_snap_shot_width: int, snap_shot_width: int
    ) -> np.ndarray:
        """
        Processes the given signal into chunks based on specified snapshot widths and calculates the average of each chunk.

        :param signal: The raw signal data as a numpy array.
        :param no_snap_shot_width: Width of non-snapshot portion of the signal in samples.
        :param snap_shot_width: Width of the snapshot portion of the signal in samples.
        :return: A numpy array containing the mean value of each snapshot segment.
        """
        block_num = int(len(signal) / (no_snap_shot_width + snap_shot_width))
        c = np.zeros(block_num)
        for i in range(block_num):
            c[i] = np.mean(
                signal[
                    i
                    * (no_snap_shot_width + snap_shot_width) : i
                    * (no_snap_shot_width + snap_shot_width)
                    + snap_shot_width
                ]
            )
        return c

    def gen_key(c: np.ndarray, rel_thresh: float, abs_thresh: float) -> str:
        """
        Generates a key based on the relative and absolute thresholds applied to the processed signal.

        :param c: The processed signal data from `signal_preprocessing`.
        :param rel_thresh: The relative threshold for generating bits.
        :param abs_thresh: The absolute threshold for generating bits.
        :return: A binary string representing the generated key.
        """
        bits = ""

        for i in range(len(c) - 1):
            feature1 = np.abs((c[i] / c[i - 1]) - 1)
            feature2 = np.abs(c[i] - c[i - 1])
            if feature1 > rel_thresh and feature2 > abs_thresh:
                bits += "1"
            else:
                bits += "0"
        return bits

    def miettinen_algo(
        x: np.ndarray, f: int, w: int, rel_thresh: float, abs_thresh: float
    ) -> bytes:
        """
        Main algorithm for key generation using signal processing and threshold-based key derivation.

        :param x: The raw signal data.
        :param f: The frame rate factor, used to calculate the window size for the signal processing.
        :param w: The window size factor, used alongside the frame rate to define the granularity of signal analysis.
        :param rel_thresh: The relative threshold for feature extraction in key generation.
        :param abs_thresh: The absolute threshold for feature extraction in key generation.
        :return: A byte string of the generated key.
        """

        def bitstring_to_bytes(s):
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        signal = MiettinenProcessing.signal_preprocessing(x, f, w)
        key = MiettinenProcessing.gen_key(signal, rel_thresh, abs_thresh)
        
        return bitstring_to_bytes(key)
