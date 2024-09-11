from typing import List


import numpy as np
from scipy.fft import rfft
import scipy

class SchurmannProcessing:
    def sigs_algo(x1: List[float], window_len: int = 10000, bands: int = 1000) -> bytes:
        """
        Signal processing algorithm that computes a bit string based on the energy difference between bands of Fourier transforms.

        :param x1: Input signal array.
        :param window_len: Length of the window for Fourier transform.
        :param bands: Number of frequency bands to consider.
        :return: A bit string converted to bytes based on the energy differences.
        """

        def bitstring_to_bytes(s: str) -> bytes:
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        FFTs = []
        # from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x1)  # renamed x to x1 because x was undefined

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

        # count = 0
        for i in range(1, len(FFTs)):
            for j in range(0, len(bands_lst[i]) - 1):
                if E[(i, j)] - E[(i, j + 1)] - (E[(i - 1, j)] - E[(i - 1, j + 1)]) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)

    def zero_out_antialias_sigs_algo(
        x1: List[float],
        antialias_freq: float,
        sampling_freq: float,
        window_len: int = 10000,
        bands: int = 1000,
    ) -> bytes:
        """
        Similar to sigs_algo but zeroes out frequencies above the anti-aliasing frequency before computing the bit string.

        :param x1: Input signal array.
        :param antialias_freq: Anti-aliasing frequency threshold.
        :param sampling_freq: Sampling frequency of the input signal.
        :param window_len: Length of the window for Fourier transform.
        :param bands: Number of frequency bands to consider.
        :return: A bit string converted to bytes.
        """

        def bitstring_to_bytes(s: str) -> bytes:
            return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

        FFTs = []
        # from scipy.fft import fft, fftfreq, ifft, irfft, rfft

        if window_len == 0:
            window_len = len(x1)  # renamed x to x1 because x is not defined

        freq_bin_len = (sampling_freq / 2) / (int(window_len / 2) + 1)
        antialias_bin = int(antialias_freq / freq_bin_len)

        x = np.array(x1.copy())
        wind = scipy.signal.windows.hann(window_len)
        for i in range(0, len(x), window_len):
            if len(x[i : i + window_len]) < window_len:
                wind = scipy.signal.windows.hann(len(x[i:i+window_len]))
                x[i : i + window_len] = x[i : i + window_len]  * wind
            else:
                x[i : i + window_len] = x[i : i + window_len]  * wind

            fft_row = abs(rfft(x[i : i + window_len]))
            FFTs.append(fft_row[:antialias_bin])
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
                if (E[(i, j)] - E[(i, j + 1)]) - (
                    E[(i - 1, j)] - E[(i - 1, j + 1)]
                ) > 0:
                    bs += "1"
                else:
                    bs += "0"
        return bitstring_to_bytes(bs)
