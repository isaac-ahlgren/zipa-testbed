import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(1, os.path.join(os.getcwd(), "eval"))

from utils import load_bytes  # noqa: E402

# Define plotting directories for both datasets
SCHURMANN_PLOTTING_DIR = "./eval/plot_scripts/plot_data/schurmann_real_full_week"
MIETTINEN_PLOTTING_DIR = "./eval/plot_scripts/plot_data/miettinen_real_full_week"

# Ensure the directories exist
if not os.path.exists(SCHURMANN_PLOTTING_DIR):
    os.makedirs(SCHURMANN_PLOTTING_DIR)

if not os.path.exists(MIETTINEN_PLOTTING_DIR):
    os.makedirs(MIETTINEN_PLOTTING_DIR)

# Device groups for Schurmann
SCHURMANN_DEVICE_GROUPS = [
    ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
    ("10.0.0.234", "10.0.0.239", "10.0.0.237"),
    ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
    ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
    ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
    ("10.0.0.238", "10.0.0.228", "10.0.0.239"),
]

# Device groups for Miettinen
MIETTINEN_DEVICE_GROUPS = [
    ("10.0.0.227", "10.0.0.229", "10.0.0.237"),
    ("10.0.0.234", "10.0.0.239", "10.0.0.237"),
    ("10.0.0.231", "10.0.0.232", "10.0.0.239"),
    ("10.0.0.235", "10.0.0.237", "10.0.0.239"),
    ("10.0.0.233", "10.0.0.236", "10.0.0.239"),
    ("10.0.0.238", "10.0.0.228", "10.0.0.239"),
]


def get_block_err(bits1, bits2, block_size):
    total = 0
    num_of_blocks = len(bits1) // block_size
    if len(bits1) != 0 and len(bits2) != 0:
        for i in range(0, len(bits1), block_size):
            sym1 = bits1[i : i + block_size]
            sym2 = bits2[i : i + block_size]
            if sym1 != sym2:
                total += 1
        bit_err = (total / num_of_blocks) * 100
    else:
        bit_err = 50
    return bit_err


def process_device_data(week_data, block_size):
    if len(week_data) != 3:
        raise ValueError("Expected 3 devices in week_data")

    legit_comparison = []
    adversarial_comparison = []

    for hour in range(168):
        print(f"Processing data for hour {hour + 1}...")

        hour_data = [week_data[device][hour] for device in range(3)]

        legit_ber = get_block_err(hour_data[0], hour_data[1], block_size)
        legit_comparison.append(legit_ber)

        adv_ber = get_block_err(hour_data[0], hour_data[2], block_size)
        adversarial_comparison.append(adv_ber)

    return legit_comparison, adversarial_comparison


def plot_ber_comparisons(
    legit_comparison, adv_comparison, custom_plot_filename, plotting_dir
):
    hours = np.arange(1, 169)

    plt.figure(figsize=(10, 7))
    plt.plot(hours, legit_comparison, marker="o", label="Legitimate", color="blue")
    plt.plot(hours, adv_comparison, marker="o", label="Adversarial", color="red")
    plt.title("Hourly BER: Legitimate vs Adversarial Comparison (Week)")
    plt.xlabel("Hours of the Week")
    plt.ylabel("Average Bit Error Rate (%)")
    plt.xticks(
        np.arange(0, 169, 24),
        [
            "Mon 0",
            "Mon 24",
            "Tue 48",
            "Wed 72",
            "Thu 96",
            "Fri 120",
            "Sat 144",
            "Sun 168",
        ],
    )
    plt.grid()
    plt.legend()
    plt.tight_layout()

    custom_plot_path = os.path.abspath(os.path.join(plotting_dir, custom_plot_filename))
    print(f"Saving combined comparison plot to: {custom_plot_path}")
    plt.savefig(custom_plot_path)
    plt.show()


def save_to_csv(legit_comparison, adv_comparison, custom_csv_filename, plotting_dir):
    hours = np.arange(1, 169)
    data = {
        "Hour": hours,
        "Legitimate_Comparison": legit_comparison,
        "Adversarial_Comparison": adv_comparison,
    }
    df = pd.DataFrame(data)

    custom_csv_path = os.path.abspath(os.path.join(plotting_dir, custom_csv_filename))
    df.to_csv(custom_csv_path, index=False)
    print(f"BER data saved to: {custom_csv_path}")


# Main execution loop to process device groupings for both datasets
if __name__ == "__main__":
    block_size = 1  # Define the block size

    # Process Schurmann data
    for group in SCHURMANN_DEVICE_GROUPS:
        week_data = [
            load_bytes(
                f"local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/schurmann_real_eval_full_two_weeks_{group[0]}_bits.txt"
            ),
            load_bytes(
                f"local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/schurmann_real_eval_full_two_weeks_{group[1]}_bits.txt"
            ),
            load_bytes(
                f"local_data/schurmann_real_full/schurmann_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/schurmann_real_eval_full_two_weeks_{group[2]}_bits.txt"
            ),
        ]

        legit_comparison, adv_comparison = process_device_data(week_data, block_size)

        # Generate plot and CSV filenames dynamically for Schurmann
        plot_filename = (
            f"{group[0]}_{group[1]}_{group[2]}_ber_comparison_plot_schurmann_week.pdf"
        )
        csv_filename = (
            f"{group[0]}_{group[1]}_{group[2]}_ber_comparison_data_schurmann_week.csv"
        )

        # Plot and save Schurmann comparisons
        plot_ber_comparisons(
            legit_comparison,
            adv_comparison,
            custom_plot_filename=plot_filename,
            plotting_dir=SCHURMANN_PLOTTING_DIR,
        )
        save_to_csv(
            legit_comparison,
            adv_comparison,
            custom_csv_filename=csv_filename,
            plotting_dir=SCHURMANN_PLOTTING_DIR,
        )

    # Process Miettinen data
    for group in MIETTINEN_DEVICE_GROUPS:
        week_data = [
            load_bytes(
                f"local_data/miettinen_real_full/miettinen_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/miettinen_real_eval_full_two_weeks_{group[0]}_bits.txt"
            ),
            load_bytes(
                f"local_data/miettinen_real_full/miettinen_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/miettinen_real_eval_full_two_weeks_{group[1]}_bits.txt"
            ),
            load_bytes(
                f"local_data/miettinen_real_full/miettinen_real_eval_full_two_weeks_{group[0]}_{group[1]}_{group[2]}/miettinen_real_eval_full_two_weeks_{group[2]}_bits.txt"
            ),
        ]

        legit_comparison, adv_comparison = process_device_data(week_data, block_size)

        # Generate plot and CSV filenames dynamically for Miettinen
        plot_filename = (
            f"{group[0]}_{group[1]}_{group[2]}_ber_comparison_plot_miettinen_week.pdf"
        )
        csv_filename = (
            f"{group[0]}_{group[1]}_{group[2]}_ber_comparison_data_miettinen_week.csv"
        )

        # Plot and save Miettinen comparisons
        plot_ber_comparisons(
            legit_comparison,
            adv_comparison,
            custom_plot_filename=plot_filename,
            plotting_dir=MIETTINEN_PLOTTING_DIR,
        )
        save_to_csv(
            legit_comparison,
            adv_comparison,
            custom_csv_filename=csv_filename,
            plotting_dir=MIETTINEN_PLOTTING_DIR,
        )
