from typing import Any, Callable, List, Tuple
from multiprocessing import Process

from eval_tools import Signal_Buffer, calc_all_bits, calc_all_events, cmp_bits, events_cmp_bits
from signal_file import Signal_File


class Evaluator:
    def __init__(self, bit_gen_algo_wrapper: Callable[[Any], List[bytes]], random_parameter_func=None, logging_func=None, event_driven=False):
        """
        Initialize the Evaluator with a specific bit generation algorithm.

        :param bit_gen_algo_wrapper: A function that takes a signal and returns a list of bits.
        """
        self.bit_gen_algo_wrapper = bit_gen_algo_wrapper
        self.random_parameter_func = random_parameter_func
        self.logging_func = logging_func
        self.event_driven = event_driven
        self.legit_bits1 = []
        self.legit_bits2 = []
        self.adv_bits = []

    def evaluate_controlled_signals(self, signals: Tuple[Any, Any, Any], trials: int, *argv: Any) -> None:
        """
        Evaluate the signals over a specified number of trials to generate cryptographic bits.

        :param signals: A tuple containing three signal sources (legit_signal1, legit_signal2, adv_signal).
        :param trials: The number of trials to perform bit generation.
        """
        legit_signal1, legit_signal2, adv_signal = signals
        for i in range(trials):
            bits1 = self.bit_gen_algo_wrapper(legit_signal1, *argv)
            self.legit_bits1.append(bits1)

            bits2 = self.bit_gen_algo_wrapper(legit_signal2, *argv)
            self.legit_bits2.append(bits2)

            adv_bits = self.bit_gen_algo_wrapper(adv_signal, *argv)
            self.adv_bits.append(adv_bits)

            if isinstance(legit_signal1, Signal_Buffer) or isinstance(
                legit_signal1, Signal_File
            ):
                legit_signal1.sync(legit_signal2)

    def cmp_collected_bits(
        self, key_length: int
    ) -> Tuple[List[float], List[float]]:
        """
        Compare bit errors using a specified comparison function and key length.

        :param func: A function that compares two lists of bits and returns a bit error rate.
        :param key_length: The length of the key used in the comparison.
        :return: A tuple containing lists of bit error rates for legitimate and adversary bits.
        """
        if self.event_driven:
            cmp = events_cmp_bits
        else:
            cmp = cmp_bits

        legit_bit_errs = []
        adv_bit_errs = []
        for i in range(len(self.legit_bits1)):
            legit_bit_err = cmp(self.legit_bits1[i], self.legit_bits2[i], key_length)
            adv_bit_err = cmp(self.legit_bits1[i], self.adv_bits[i], key_length)
            legit_bit_errs.append(legit_bit_err)
            adv_bit_errs.append(adv_bit_err)
        return legit_bit_errs, adv_bit_errs
    
    def reset_bits_lists(self):
        del self.legit_bits1
        del self.legit_bits2
        del self.adv_bits
        self.legit_bits1 = []
        self.legit_bits2 = []
        self.adv_bits = []

    def evaluate_device_non_ed(self, signal: Signal_File, params: Tuple):
        return calc_all_bits(signal, self.bit_gen_algo_wrapper, *params)

    def evaluate_device_ed(self, signal: Signal_File, params: Tuple):
        return calc_all_events(signal, self.bit_gen_algo_wrapper, *params)

    def fuzzing_func(self, signal, params):
        if self.event_driven:
                outcome = self.evaluate_device_ed(signal, params)
        else:
                outcome = self.evaluate_device_non_ed(signal, params)
        self.logging_func(outcome, *params)

    def fuzzing_single_threaded(self, signals, params):
        for signal in signals:
            self.fuzzing_func(signal, params)

    def fuzzing_multithreaded(self, signals, params):
        threads = []
        for signal in signals:
            p = Process(target=self.fuzzing_func, args=(signal, params))
            p.start()
            threads.append(p)

        for thread in threads:
            thread.join()

    def fuzzing_evaluation(self, signals, number_of_choices, multithreaded=True) -> None:
        for i in range(number_of_choices):
            params = self.random_parameter_func()
            if multithreaded:
                self.fuzzing_multithreaded(signals, params)
            else:
                self.fuzzing_single_threaded(signals, params)
