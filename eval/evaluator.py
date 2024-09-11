from multiprocessing import Process
from typing import Any, Callable, List, Tuple
import os

from eval_tools import (
    calc_all_bits,
    calc_all_events,
    cmp_bits,
    events_cmp_bits,
    gen_id,
    log_bytes,
)
from signal_file_interface import Signal_File_Interface


class Evaluator:
    def __init__(
        self,
        bit_gen_algo_wrapper: Callable[[Any], List[bytes]],
        random_parameter_func=None,
        parameter_log_func=None,
        event_driven=False,
    ):
        """
        Initialize the Evaluator with a specific bit generation algorithm.

        :param bit_gen_algo_wrapper: A function that takes a signal and returns a list of bits.
        """
        self.bit_gen_algo_wrapper = bit_gen_algo_wrapper
        self.random_parameter_func = random_parameter_func
        self.parameter_log_func = parameter_log_func
        self.event_driven = event_driven
        self.legit_bits1 = []
        self.legit_bits2 = []
        self.adv_bits = []

    def evaluate_controlled_signals(
        self, signals: Tuple[Any, Any, Any], trials: int, *argv: Any
    ) -> None:
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

            if isinstance(legit_signal1, Signal_File_Interface) or isinstance(
                legit_signal1, Signal_File_Interface
            ):
                legit_signal1.sync(legit_signal2)

    def cmp_collected_bits(self, key_length: int) -> Tuple[List[float], List[float]]:
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

    def evaluate_device_non_ed(self, signal: Signal_File_Interface, params: Tuple):
        return calc_all_bits(signal, self.bit_gen_algo_wrapper, *params)

    def evaluate_device_ed(self, signal: Signal_File_Interface, params: Tuple):
        return calc_all_events(signal, self.bit_gen_algo_wrapper, *params)

    def fuzzing_func(self, signal, key_length, file_stub, params):
        if self.event_driven:
            outcome = self.evaluate_device_ed(signal, params)
        else:
            outcome = self.evaluate_device_non_ed(signal, params)
        
        file_stub = file_stub + "_" + signal.get_id()
        log_bytes(file_stub, outcome, key_length)
        signal.reset()

    def fuzzing_single_threaded(self, signals, key_length, file_stub, params):
        for signal in signals:
            self.fuzzing_func(signal, key_length, file_stub, params)

    def fuzzing_multithreaded(self, signals, key_length, file_stub, params):
        threads = []
        for signal in signals:
            p = Process(target=self.fuzzing_func, args=(signal, key_length, file_stub, params))
            p.start()
            threads.append(p)

        for thread in threads:
            thread.join()

    def fuzzing_evaluation(
        self, signals, number_of_choices, key_length, fuzzing_dir, fuzzing_file_stub, multithreaded=True
    ) -> None:
        for i in range(number_of_choices):
            params = self.random_parameter_func()
            choice_id = gen_id()
            choice_file_stub = f"{fuzzing_file_stub}_id{choice_id}"
            file_dir = f"{fuzzing_dir}/{choice_file_stub}"
            if not os.path.isdir(file_dir):
                os.mkdir(file_dir)
            file_stub = file_dir + "/" + choice_file_stub
            self.parameter_log_func(params, file_stub)
            
            if multithreaded:
                self.fuzzing_multithreaded(signals, key_length, file_stub, params)
            else:
                self.fuzzing_single_threaded(signals, key_length, file_stub, params)
