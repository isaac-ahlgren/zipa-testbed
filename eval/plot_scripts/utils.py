import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(1, os.getcwd() + "/..")  # Gives us path to eval_tools.py
from eval_tools import load_events, load_parameters  # noqa: E402

PLOTTING_DIR = "./plot_data"


def load_bytes(byte_file):
    with open(byte_file, "r") as file:
        output = file.readlines()
    output = [bits.strip() for bits in output]
    return output


def parse_eval_directory(data_dir, file_stub, parse_string="bits"):
    output = []
    files = glob.glob(f"{file_stub}*", root_dir=f"{data_dir}/{file_stub}")
    max_key_length = 0
    for dir_name in files:
        dir_content = dict()

        param_file = f"{data_dir}/{file_stub}/{dir_name}/{dir_name}_params.csv"
        params = load_parameters(param_file)
        dir_content["params"] = params

        data_files = glob.glob(
            f"{dir_name}*_{parse_string}.txt",
            root_dir=f"{data_dir}/{file_stub}/{dir_name}",
        )
        for df in data_files:
            data = load_bytes(f"{data_dir}/{file_stub}/{dir_name}/{df}")

            idi = df.find("_id") + 1
            i1 = df.find("_", idi, len(df)) + 1
            i2 = df.find(".txt")
            signal_id = df[i1:i2]
            dir_content[signal_id] = data
        output.append(dir_content)
        if len(dir_content.keys()) > max_key_length:
            max_key_length = len(dir_content.keys())

    filtered_output = []
    for content in output:
        if len(content.keys()) == max_key_length:
            filtered_output.append(content)

    return filtered_output


def directly_parse_eval_directory_event_num(data_dir, file_stub):
    output = []
    files = glob.glob(f"{file_stub}*", root_dir=f"{data_dir}/{file_stub}")
    max_key_length = 0
    for dir_name in files:
        dir_content = dict()

        param_file = f"{data_dir}/{file_stub}/{dir_name}/{dir_name}_params.csv"
        params = load_parameters(param_file)
        dir_content["params"] = params

        data_files = glob.glob(
            f"{dir_name}*_time_stamps.csv",
            root_dir=f"{data_dir}/{file_stub}/{dir_name}",
        )
        for df in data_files:
            data = load_events(f"{data_dir}/{file_stub}/{dir_name}/{df}")
            data_len = len(data)
            del data

            idi = df.find("_id") + 1
            i1 = df.find("_", idi, len(df)) + 1
            i2 = df.find(".csv")
            signal_id = df[i1:i2]
            dir_content[signal_id] = data_len
        output.append(dir_content)
        if len(dir_content.keys()) > max_key_length:
            max_key_length = len(dir_content.keys())

    filtered_output = []
    for content in output:
        if len(content.keys()) == max_key_length:
            filtered_output.append(content)

    return filtered_output


def get_block_err(bits1, bits2, block_size):
    total = 0
    num_of_blocks = len(bits1) // block_size
    if len(bits1) != 0 and len(bits2) != 0:
        for i in range(0, len(bits1), block_size):
            sym1 = bits1[i * block_size : (i + 1) * block_size]
            sym2 = bits2[i * block_size : (i + 1) * block_size]
            if sym1 != sym2:
                total += 1
        bit_err = (total / num_of_blocks) * 100
    else:
        bit_err = 50
    return bit_err


def cmp_byte_list(byte_list1, byte_list2, block_size):
    block_err_list = []
    for b1, b2 in zip(byte_list1, byte_list2):
        err_rate = get_block_err(b1, b2, block_size)
        block_err_list.append(err_rate)
    return block_err_list


def get_avg_ber_list(byte_list1, byte_list2):
    avg_ber_list = []
    for contents1, contents2 in zip(byte_list1, byte_list2):
        ber_list = cmp_byte_list(contents1, contents2, 1)
        if len(ber_list) != 0:
            avg_ber = np.mean(ber_list)
        else:
            avg_ber = 50
        avg_ber_list.append(avg_ber)
    return avg_ber_list


def get_min_entropy_list(byte_list, key_length, symbol_size):
    min_entropy_list = []
    for contents in byte_list:
        min_entropy = get_min_entropy(contents, key_length, symbol_size)
        min_entropy_list.append(min_entropy)
    return min_entropy_list


def extract_from_contents(contents, key_word):
    extracted_content = []
    for content in contents:
        if key_word in content:
            extracted_content.append(content[key_word])
    return extracted_content


def get_min_entropy(bits, key_length: int, symbol_size: int) -> float:
    """
    Calculate the minimum entropy of a list of bitstrings based on symbol size.

    :param bits: List of bitstrings.
    :param key_length: The total length of each bitstring.
    :param symbol_size: The size of each symbol in bits.
    :return: The minimum entropy observed across all symbols.
    """
    arr = []
    for b in bits:
        for i in range(0, key_length // symbol_size, symbol_size):
            symbol = b[i * symbol_size : (i + 1) * symbol_size]
            arr.append(int(symbol, 2))

    hist, bin_edges = np.histogram(arr, bins=2**symbol_size)
    pdf = hist / sum(hist)
    max_prob = np.max(pdf)
    return -np.log2(max_prob)


def condense_list(ziped_list):
    outcome = []

    total_ber = dict()
    count = dict()
    for ber, param in ziped_list:
        if param in total_ber:
            count[param] += 1
            total_ber[param] += ber
        else:
            count[param] = 1
            total_ber[param] = ber

    for param in total_ber.keys():
        ber = total_ber[param] / count[param]
        outcome.append((ber, param))

    return outcome


def parameter_plot(
    ber_list,
    param_list,
    plot_name,
    param_label,
    savefig=False,
    file_name=None,
    fig_dir=None,
    range=None,
    ylabel="Bit Error",
):
    ziped_list = list(zip(ber_list, param_list))

    if range is not None:
        ziped_list = [x for x in ziped_list if range[0] <= x[1] <= range[1]]

    ziped_list = condense_list(ziped_list)

    ziped_list.sort(key=lambda x: x[1])

    ord_ber_list = [x[0] for x in ziped_list]
    ord_param_list = [x[1] for x in ziped_list]

    plt.plot(ord_param_list, ord_ber_list)
    plt.title(plot_name)
    plt.xlabel(param_label)
    plt.ylabel(ylabel)
    if savefig:
        plt.savefig(fig_dir + "/" + file_name + ".pdf")
        plt.clf()
        fig_data_name = fig_dir + "/" + file_name + ".csv"
        df = pd.DataFrame({"x_axis": ord_param_list, "y_axis": ord_ber_list})
        df.to_csv(fig_data_name)
    else:
        plt.show()


def make_plot_dir(dir_name):
    dir = PLOTTING_DIR + "/" + dir_name
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir


def find_best_parameter_choice(legit_ber, adv_ber, params, file_stub, fig_dir):
    best_score = 200
    best_legit_ber = None
    best_adv_ber = None
    best_param = None
    for legit, adv, param in zip(legit_ber, adv_ber, params):
        adv_ber_score = 2 * abs(50 - adv)
        score = adv_ber_score + legit
        if score < best_score:
            best_param = param
            best_legit_ber = legit
            best_adv_ber = adv
            best_score = score

    if best_param is not None:
        copy_best_param = best_param.copy()
        copy_best_param["adv_ber"] = [best_adv_ber]
        copy_best_param["legit_ber"] = [best_legit_ber]

        file_name = fig_dir + "/" + file_stub + "_bestparam.csv"
        df = pd.DataFrame(copy_best_param)
        df.to_csv(file_name)
