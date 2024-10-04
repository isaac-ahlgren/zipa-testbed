import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert the path to the parent directory of eval_tools.py
sys.path.insert(1, os.path.join(os.getcwd(), "eval"))

from eval_tools import load_parameters  # noqa: E402

PLOTTING_DIR = "./plot_data"

# Update DEVICE_PAIRS to reflect the 3 devices you're using
DEVICE_PAIRS = [
    ("10.0.0.227", "10.0.0.229", "10.0.0.237")  # Only using 3 devices
]


def load_bytes(byte_file):
    with open(byte_file, "r") as file:
        output = file.readlines()
    output = [bits.strip() for bits in output]
    return output


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
    if len(bits1) != 0 and len(bits2) != 0:
        for i in range(0, len(bits1), block_size):
            sym1 = bits1[i * block_size: (i + 1) * block_size]
            sym2 = bits2[i * block_size: (i + 1) * block_size]
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


def extract_from_contents(contents, key_word):
    extracted_content = []
    for content in contents:
        extracted_content.append(content[key_word])
    return extracted_content


def get_min_entropy(bits, key_length: int, symbol_size: int) -> float:
    arr = []
    for b in bits:
        for i in range(0, key_length // symbol_size, symbol_size):
            symbol = b[i * symbol_size: (i + 1) * symbol_size]
            arr.append(int(symbol, 2))

    hist, bin_edges = np.histogram(arr, bins=2 ** symbol_size)
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


def bit_err_vs_parameter_plot(ber_list, param_list, plot_name, param_label, savefig=False, file_name=None, fig_dir=None, range=None):
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


def process_device_data(week_data, block_size):
    """
    Processes weekly data for devices organized into pairs defined in DEVICE_PAIRS.
    
    :param week_data: A list of 3 elements, each containing 7 days of byte data for a device.
    :param block_size: The size of blocks for comparison.
    :return: A dictionary of average BER for each group comparison for each day.
    """
    # Ensure there are exactly 3 devices
    if len(week_data) != 3:
        raise ValueError("Expected 3 devices in week_data")
    
    # Create a mapping of IP addresses to their index in the week_data list
    device_map = {
        "10.0.0.227": 0,
        "10.0.0.229": 1,
        "10.0.0.237": 2
    }

    daily_avg_ber = {
        'Device Comparisons': [],
    }  # Dictionary to hold average BER for comparisons

    # Loop through each day
    for day in range(7):
        print(f"Processing data for day {day + 1}...")
        
        # Extract byte data for the current day from each device
        day_data = [week_data[device][day] for device in range(3)]
        
        # Calculate BER for each pair comparison for the current day
        avg_ber = {}
        for device1, device2, reference_device in DEVICE_PAIRS:
            try:
                index1 = device_map[device1]
                index2 = device_map[device2]
                avg_ber[f'{device1} vs {device2} (ref: {reference_device})'] = get_block_err(day_data[index1], day_data[index2], block_size)
            except KeyError as e:
                print(f"Error: Device {e} not found in device map")
                continue

        # Store the average BER for the day in the dictionary
        daily_avg_ber['Device Comparisons'].append(avg_ber)

    return daily_avg_ber


def plot_daily_ber(daily_avg_ber):
    """
    Plots the daily average Bit Error Rate (BER) for each device comparison.

    :param daily_avg_ber: A dictionary of average BER for each pair of devices.
    """
    days = np.arange(1, 8)  # Days of the week from 1 to 7
    
    # Prepare data for plotting
    comparisons = daily_avg_ber['Device Comparisons']
    avg_ber_values = {comparison: [] for comparison in comparisons[0].keys()}
    
    for day_data in comparisons:
        for comparison, ber in day_data.items():
            avg_ber_values[comparison].append(ber)

    plt.figure(figsize=(12, 6))
    
    for comparison, ber_values in avg_ber_values.items():
        plt.plot(days, ber_values, marker='o', label=comparison)

    plt.title('Daily Bit Error Rate (BER) Comparison')
    plt.xlabel('Days of the Week')
    plt.ylabel('Average Bit Error Rate (%)')
    plt.xticks(days, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Ensure PLOTTING_DIR exists
    if not os.path.exists(PLOTTING_DIR):
        os.makedirs(PLOTTING_DIR)
        print(f"Created directory: {PLOTTING_DIR}")

    # Save the plot
    full_path = os.path.abspath(os.path.join(PLOTTING_DIR, 'daily_ber_comparison.pdf'))
    print(f"Saving plot to: {full_path}")
    plt.savefig(full_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Loading the new bit files into the week_data list
    week_data = [
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.227_bits.txt"),
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.229_bits.txt"),
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.237_bits.txt")
    ]

    block_size = 1  # Define the block size
    daily_avg_ber = process_device_data(week_data, block_size)
    plot_daily_ber(daily_avg_ber)
