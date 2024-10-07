import os
from multiprocessing import Process
from typing import Any, Callable, List, Tuple

from eval_tools import (
    calc_all_bits,
    cmp_bits,
    events_cmp_bits,
    gen_id,
    log_bit_gen_outcomes,
    log_event_gen_outcomes,
    log_seed,
)
from signal_file_interface import Signal_File_Interface


class Evaluator:
    def __init__(
        self,
        func: Callable[[Any], List[bytes]],
        random_parameter_func=None,
        parameter_log_func=None,
        event_gen=False,
        log_seed=False,
    ):
        """
        Initialize the Evaluator with a specific bit generation algorithm.

        :param bit_gen_algo_wrapper: A function that takes a signal and returns a list of bits.
        """
        self.func = func
        self.random_parameter_func = random_parameter_func
        self.parameter_log_func = parameter_log_func
        self.event_gen = event_gen
        self.log_seed = log_seed
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
            bits1 = self.func(legit_signal1, *argv)
            self.legit_bits1.append(bits1)

            bits2 = self.func(legit_signal2, *argv)
            self.legit_bits2.append(bits2)

            adv_bits = self.func(adv_signal, *argv)
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
        if self.event_gen:
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

    def evaluate_device_bit_gen(self, signal: Signal_File_Interface, params: Tuple):
        return calc_all_bits(signal, self.func, *params)

    def eval_event_gen_func(self, signal: Signal_File_Interface, key_length, file_stub, params):
        event_timestamps = self.func(signal, *params)
        file_stub = file_stub + "_" + signal.get_id()

        log_event_gen_outcomes(file_stub, event_timestamps)

        if self.log_seed:
            log_seed(file_stub, signal.seed)

    def eval_bit_gen_func(self, signal, key_length, file_stub, params):
        outcome, extras = self.evaluate_device_bit_gen(signal, params)

        file_stub = file_stub + "_" + signal.get_id()
        log_bit_gen_outcomes(file_stub, outcome, extras, key_length)
        signal.reset()

    def eval_func(self, *params):
        if self.event_gen:
            self.eval_event_gen_func(*params)
        else:
            self.eval_bit_gen_func(*params)

    def eval_single_threaded(self, signals, key_length, file_stub, params):
        for signal in signals:
            self.eval_func(signal, key_length, file_stub, params)

    def eval_multithreaded(self, signals, key_length, file_stub, params):
        threads = []
        for signal in signals:
            p = Process(
                target=self.eval_func, args=(signal, key_length, file_stub, params)
            )
            p.start()
            threads.append(p)

        for thread in threads:
            thread.join()

    def best_parameter_evaluation(self, group_signals, group_params, key_length, dir, file_stub):
        threads = []
        for signal_group, params in zip(group_signals, group_params):
            id1 = signal_group[0].get_id()
            id2 = signal_group[1].get_id()
            id3 = signal_group[2].get_id()
            dir_path = f"{dir}/{file_stub}_{id1}_{id2}_{id3}"
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            stub = f"{dir_path}/{file_stub}"
            for signal in signal_group:
                p = Process(target=self.eval_func, args=(signal, key_length, stub, params))
                p.start()
                threads.append(p)
        
        for thread in threads:
            thread.join()

    def fuzzing_evaluation(
        self,
        signals,
        number_of_choices,
        key_length,
        fuzzing_dir,
        fuzzing_file_stub,
        multithreaded=True,
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
                self.eval_multithreaded(signals, key_length, file_stub, params)
            else:
                self.eval_single_threaded(signals, key_length, file_stub, params)

