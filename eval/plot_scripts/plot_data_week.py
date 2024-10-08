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


def process_device_data(week_data, block_size):
    """
    Processes weekly data for devices organized into pairs defined in DEVICE_PAIRS.
    
    :param week_data: A list of 3 elements, each containing 7 days of byte data for a device.
    :param block_size: The size of blocks for comparison.
    :return: Two lists containing average BER for legitimate and adversarial comparisons for each day.
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

    legit_comparison = []
    adversarial_comparison = []

    # Loop through each day
    for day in range(7):
        print(f"Processing data for day {day + 1}...")

        # Extract byte data for the current day from each device
        day_data = [week_data[device][day] for device in range(3)]
        
        # Calculate BER for legitimate comparison (227 vs 229)
        legit_ber = get_block_err(day_data[device_map["10.0.0.227"]], 
                                  day_data[device_map["10.0.0.229"]], block_size)
        legit_comparison.append(legit_ber)

        # Calculate BER for adversarial comparison (227 vs 237)
        adv_ber = get_block_err(day_data[device_map["10.0.0.227"]], 
                                day_data[device_map["10.0.0.237"]], block_size)
        adversarial_comparison.append(adv_ber)

    return legit_comparison, adversarial_comparison


def plot_ber_comparisons(legit_comparison, adv_comparison):
    """
    Plots the daily average Bit Error Rate (BER) for legitimate and adversarial comparisons separately.

    :param legit_comparison: A list of BER values for legitimate comparison (227 vs 229).
    :param adv_comparison: A list of BER values for adversarial comparison (227 vs 237).
    """
    days = np.arange(1, 8)  # Days of the week from 1 to 7

    # Plot legitimate comparison
    plt.figure(figsize=(8, 6))
    plt.plot(days, legit_comparison, marker='o', label="10.0.0.227 vs 10.0.0.229", color='blue')
    plt.title('Daily BER: Legitimate Comparison (10.0.0.227 vs 10.0.0.229)')
    plt.xlabel('Days of the Week')
    plt.ylabel('Average Bit Error Rate (%)')
    plt.xticks(days, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid()
    plt.tight_layout()

    legit_plot_path = os.path.abspath(os.path.join(PLOTTING_DIR, 'legit_comparison_ber.pdf'))
    print(f"Saving legitimate comparison plot to: {legit_plot_path}")
    plt.savefig(legit_plot_path)
    plt.show()

    # Plot adversarial comparison
    plt.figure(figsize=(8, 6))
    plt.plot(days, adv_comparison, marker='o', label="10.0.0.227 vs 10.0.0.237", color='red')
    plt.title('Daily BER: Adversarial Comparison (10.0.0.227 vs 10.0.0.237)')
    plt.xlabel('Days of the Week')
    plt.ylabel('Average Bit Error Rate (%)')
    plt.xticks(days, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid()
    plt.tight_layout()

    adv_plot_path = os.path.abspath(os.path.join(PLOTTING_DIR, 'adversarial_comparison_ber.pdf'))
    print(f"Saving adversarial comparison plot to: {adv_plot_path}")
    plt.savefig(adv_plot_path)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Loading the new bit files into the week_data list
    week_data = [
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.227_10.0.0.229_10.0.0.237/schurmann_real_eval_full_two_weeks_10.0.0.227_bits.txt"),
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.227_10.0.0.229_10.0.0.237/schurmann_real_eval_full_two_weeks_10.0.0.229_bits.txt"),
        load_bytes("local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_10.0.0.227_10.0.0.229_10.0.0.237/schurmann_real_eval_full_two_weeks_10.0.0.237_bits.txt")
    ]

    block_size = 1  # Define the block size
    legit_comparison, adv_comparison = process_device_data(week_data, block_size)

    # Plot separate comparisons
    plot_ber_comparisons(legit_comparison, adv_comparison)
