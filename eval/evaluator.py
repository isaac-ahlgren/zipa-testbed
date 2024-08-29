from typing import Any, Callable, List, Tuple
import numpy as np

from eval_tools import Signal_Buffer, Signal_File


class Evaluator:
    def __init__(self, bit_gen_algo_wrapper: Callable[[Any], List[bytes]]):
        """
        Initialize the Evaluator with a specific bit generation algorithm.

        :param bit_gen_algo_wrapper: A function that takes a signal and returns a list of bits.
        """
        self.bit_gen_algo_wrapper = bit_gen_algo_wrapper
        self.legit_bits1 = []
        self.legit_bits2 = []
        self.adv_bits = []
        self.legit_avg_bit_err = []
        self.adv_avg_bit_errs = []

    def evaluate(self, signals: Tuple[Any, Any, Any], trials: int, *argv: Any) -> None:
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

    def fuzzing_evaluation(self, signals: Tuple[Any, Any, Any], logging_func, random_parameter_func, cmp_func, trials_per_choice, number_of_choices, key_size) -> None:
        for i in range(number_of_choices):
            params = random_parameter_func()
            self.evaluate(signals, trials_per_choice, *params)
            legit_bit_errs, adv_bit_errs =  self.cmp_func(cmp_func, key_size)
            logging_func(legit_bit_errs, adv_bit_errs, *params)
            self.reset_bits_lists()
            signals[0].reset()
            signals[1].reset()
            signals[2].reset()

    def cmp_func(
        self, func: Callable[[List[bytes], List[bytes], int], float], key_length: int
    ) -> Tuple[List[float], List[float]]:
        """
        Compare bit errors using a specified comparison function and key length.

        :param func: A function that compares two lists of bits and returns a bit error rate.
        :param key_length: The length of the key used in the comparison.
        :return: A tuple containing lists of bit error rates for legitimate and adversary bits.
        """
        legit_bit_errs = []
        adv_bit_errs = []
        for i in range(len(self.legit_bits1)):
            legit_bit_err = func(self.legit_bits1[i], self.legit_bits2[i], key_length)
            adv_bit_err = func(self.legit_bits1[i], self.adv_bits[i], key_length)
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
