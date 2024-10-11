import glob
import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


matplotlib.use('Agg')
# Insert the path to the parent directory of eval_tools.py
sys.path.insert(1, os.path.join(os.getcwd(), "eval"))


from eval_tools import load_parameters  # noqa: E402
from utils import load_bytes


# Set PLOTTING_DIR to the correct path within ZIPA-TESTBED
PLOTTING_DIR = "./eval/plot_scripts/plot_data/schurmann_real_full_week"


# Check if the directory exists, if not, create it
if not os.path.exists(PLOTTING_DIR):
   os.makedirs(PLOTTING_DIR)


# Update DEVICE_PAIRS to reflect the 3 devices you're using
DEVICE_PAIRS = [
   ("10.0.0.227", "10.0.0.229", "10.0.0.237")  # Only using 3 devices
]


def get_block_err(bits1, bits2, block_size):
   """
   Calculates the block error rate between two bit sequences.
  
   :param bits1: Bit sequence from the first device.
   :param bits2: Bit sequence from the second device.
   :param block_size: Size of blocks for comparison.
   :return: Bit Error Rate (BER) as a percentage.
   """
   total = 0
   num_of_blocks = len(bits1) // block_size
   if len(bits1) != 0 and len(bits2) != 0:
       for i in range(0, len(bits1), block_size):
           sym1 = bits1[i: i + block_size]  # Adjusted to fit block size
           sym2 = bits2[i: i + block_size]
           if sym1 != sym2:
               total += 1
       bit_err = (total / num_of_blocks) * 100
   else:
       bit_err = 50
   return bit_err




def process_device_data(week_data, block_size):
   """
   Processes hourly data for devices over the course of a week (168 hours).
  
   :param week_data: A list of 3 elements, each containing 168 hours of byte data for a device.
   :param block_size: The size of blocks for comparison.
   :return: Two lists containing average BER for legitimate and adversarial comparisons for each hour.
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


   # Loop through each hour (168 hours in a week)
   for hour in range(168):
       print(f"Processing data for hour {hour + 1}...")


       # Extract byte data for the current hour from each device
       hour_data = [week_data[device][hour] for device in range(3)]
      
       # Calculate BER for legitimate comparison (227 vs 229)
       legit_ber = get_block_err(hour_data[device_map["10.0.0.227"]],
                                 hour_data[device_map["10.0.0.229"]], block_size)
       legit_comparison.append(legit_ber)


       # Calculate BER for adversarial comparison (227 vs 237)
       adv_ber = get_block_err(hour_data[device_map["10.0.0.227"]],
                               hour_data[device_map["10.0.0.237"]], block_size)
       adversarial_comparison.append(adv_ber)


   return legit_comparison, adversarial_comparison




def plot_ber_comparisons(legit_comparison, adv_comparison, custom_plot_filename):
   """
   Plots the hourly average Bit Error Rate (BER) for legitimate and adversarial comparisons over the week on the same graph.


   :param legit_comparison: A list of BER values for legitimate comparison (227 vs 229).
   :param adv_comparison: A list of BER values for adversarial comparison (227 vs 237).
   :param custom_plot_filename: Custom filename for saving the plot.
   """
   hours = np.arange(1, 169)  # Hours of the week from 1 to 168


   # Plot both legitimate and adversarial comparison on the same graph
   plt.figure(figsize=(10, 7))
   plt.plot(hours, legit_comparison, marker='o', label="10.0.0.227 vs 10.0.0.229 (Legitimate)", color='blue')
   plt.plot(hours, adv_comparison, marker='o', label="10.0.0.227 vs 10.0.0.237 (Adversarial)", color='red')
   plt.title('Hourly BER: Legitimate vs Adversarial Comparison (Week)')
   plt.xlabel('Hours of the Week')
   plt.ylabel('Average Bit Error Rate (%)')
   plt.xticks(np.arange(0, 169, 24),
              ['Mon 0', 'Mon 24', 'Tue 48', 'Wed 72', 'Thu 96', 'Fri 120', 'Sat 144', 'Sun 168'])  # Markers for each day
   plt.grid()
   plt.legend()
   plt.tight_layout()


   custom_plot_path = os.path.abspath(os.path.join(PLOTTING_DIR, custom_plot_filename))
   print(f"Saving combined comparison plot to: {custom_plot_path}")
   plt.savefig(custom_plot_path)
   plt.show()




def save_to_csv(legit_comparison, adv_comparison, custom_csv_filename):
   """
   Saves the BER data to a CSV file for further analysis.
  
   :param legit_comparison: A list of BER values for legitimate comparison (227 vs 229).
   :param adv_comparison: A list of BER values for adversarial comparison (227 vs 237).
   :param custom_csv_filename: Custom filename for saving the CSV data.
   """
   hours = np.arange(1, 169)  # Hours of the week from 1 to 168
   data = {
       "Hour": hours,
       "Legitimate_Comparison": legit_comparison,
       "Adversarial_Comparison": adv_comparison
   }
   df = pd.DataFrame(data)
  
   custom_csv_path = os.path.abspath(os.path.join(PLOTTING_DIR, custom_csv_filename))
   df.to_csv(custom_csv_path, index=False)
   print(f"BER data saved to: {custom_csv_path}")




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


   # Plot combined comparisons for hourly data over the week with a custom plot filename
   plot_ber_comparisons(legit_comparison, adv_comparison, custom_plot_filename="227-229-237_ber_comparison_plot_week.pdf")


   # Save the BER data to a custom CSV file
   save_to_csv(legit_comparison, adv_comparison, custom_csv_filename="227-229-237_ber_comparison_data_week.csv")