import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLOTTING_DIR = "./plot_data"

def load_bytes(byte_file):
    with open(byte_file, "r") as file:
        output = file.readlines()
    output = [bits.strip() for bits in output]
    return output


def fix_dict(d):
    new_d = dict()
    for k in d.keys():
        if k == "Unnamed: 0":
            continue
        val = d[k]
        ext_val = val[0]
        new_d[k] = ext_val
    return new_d


def load_parameters(param_file):
    df = pd.read_csv(param_file)
    df_dict = df.to_dict()
    params = fix_dict(df_dict)
    return params


def parse_eval_directory(data_dir, file_stub):
    output = []
    files = glob.glob(f"{file_stub}*", root_dir=f"{data_dir}/{file_stub}")
    max_key_length = 0
    for dir_name in files:
        dir_content = dict()

        param_file = f"{data_dir}/{file_stub}/{dir_name}/{dir_name}_params.csv"
        params = load_parameters(param_file)
        dir_content["params"] = params

        data_files = glob.glob(
            f"{dir_name}*_bits.txt", root_dir=f"{data_dir}/{file_stub}/{dir_name}"
        )
        for df in data_files:
            data = load_bytes(f"{data_dir}/{file_stub}/{dir_name}/{df}")

            idi = df.find("id")
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


def get_block_err(bits1, bits2, block_size):
    total = 0
    num_of_blocks = len(bits1) // block_size
    for i in range(0, len(bits1), block_size):
        sym1 = bits1[i * block_size : (i + 1) * block_size]
        sym2 = bits2[i * block_size : (i + 1) * block_size]
        if sym1 != sym2:
            total += 1
    return (total / num_of_blocks) * 100


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
        avg_ber = np.mean(ber_list)
        avg_ber_list.append(avg_ber)
    return avg_ber_list


def extract_from_contents(contents, key_word):
    extracted_content = []
    for content in contents:
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

def bit_err_vs_parameter_plot(ber_list, param_list, plot_name, param_label, savefig=False, file_name=None, fig_dir=None, range=None):
    ziped_list = list(zip(ber_list, param_list))

    if range is not None:
        ziped_list = [x for x in ziped_list if range[0] <= x[1] <= range[1]]

    ziped_list.sort(key=lambda x: x[1])

    ord_ber_list = [x[0] for x in ziped_list]
    ord_param_list = [x[1] for x in ziped_list]

    plt.plot(ord_param_list, ord_ber_list)
    plt.title(plot_name)
    plt.xlabel(param_label)
    plt.ylabel("Bit Error")
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

