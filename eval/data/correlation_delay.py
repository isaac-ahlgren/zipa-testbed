import numpy as np
import pandas as pd
from scipy.stats import pearsonr 

# should output a trimmed script with the best correlation immediately
# FILL OUT EVERYTHING FROM HERE
sensor = 'mic'
id_1 = ''
id_2 = ''

# format : yyyymmddhh
date = ''
# TO HERE 

file1 = f'/mnt/nas/{sensor}_id_10.0.0.{id_1}_date_{date}.csv'
file2 = f'/mnt/nas/{sensor}_id_10.0.0.{id_2}_date_{date}.csv'
trim_file = 'storage/storage_237_0826_18.csv'

filepath = pd.read_csv(trim_file, header=None)

df1 = pd.read_csv(file1, header=None)
df2 = pd.read_csv(file2, header=None)
                  
df1_size = (len(df1))
df2_size = (len(df2))

if df1_size <= df2_size:
    print('d2 is bigger')
    output = f'shifted_{sensor}_{id_2}_{date}.csv'
    smaller_df = df1
    larger_df = df2
else: 
    print("df1 is larger")
    output = f'shifted_{sensor}_{id_1}_{date}.csv'
    smaller_df = df2
    larger_df = df1

difference = (len(larger_df)) - (len(smaller_df))

for i in range(24, 15, -1):
    if difference % i == 0: 
        steps = difference / i  
        break

def find_correlation_index():
    global max_value
    global max_index
    CORR_VALS = smaller_df[0].values
    window_size = len(CORR_VALS)

    correlations = []

    for start in range(0, len(larger_df) - window_size + 1, steps):
        window = larger_df.iloc[start:start + window_size, 0].values
        corr = pearsonr(window, CORR_VALS)[0]
        correlations.append(corr)

    cdf = pd.DataFrame(correlations)

    max_value = cdf[0].max()
    max_index = cdf[0].idxmax()

    print(f'regular: max value is: {max_value} at {max_index}')

def bt_correlation_index():
    global bt_val
    global bt_index
    global bt_window_size

    backtrack_correlations = []

    # cutting off the first 200k samples
    smaller_df_trimmed = smaller_df.iloc[200000:]
    BT_CORR_VALS = smaller_df_trimmed[0].values
    bt_window_size = len(BT_CORR_VALS)
    # whyyyy are all my numbers hardcoded @_@ 
    start = len(larger_df) - 100000

    for start in range(start - bt_window_size, - 1, - steps):
        bt_window = larger_df.iloc[start:start + bt_window_size, 0].values
        bt_corr = pearsonr(bt_window, BT_CORR_VALS)[0]
        backtrack_correlations.append(bt_corr)

    bt_cdf = pd.DataFrame(backtrack_correlations)

    bt_val = bt_cdf[0].max()
    bt_index = bt_cdf[0].idxmax()

    print(f'backtrack: max value is: {bt_val} at {bt_index}')

def shift_data():
    if bt_val < max_value:
        shifted = max_index * steps

        remainder = (len(larger_df)) - (difference - shifted)
        larger_arr_trimmed = larger_df.iloc[shifted:remainder]
    
        larger_arr_trimmed.to_csv(output, index=False, header=None)
    # i literally have no idea if this works but im so braindead rn so let's test it out
    elif max_value < bt_val:
        shifted = (len(larger_df) - bt_window_size) - (bt_index * steps)
        remainder = shifted + bt_window_size

        larger_arr_trimmed = larger_df.iloc[shifted:remainder]
        larger_arr_trimmed.to_csv(output, index=False, header=None)

# ngl i have no idea what to do with this rn i just want to make sure all the dataframes are the same length
# and it's kinda just hardcoded there
# maybe just ignore this for now
def shorten_file():
    if (len(filepath)) > 172700000:
        kill = 172700000
        filepath_trimmed = filepath.head(kill)

        filepath_trimmed.to_csv('double_check.csv', index=False, header=None)

find_correlation_index()
bt_correlation_index()
shift_data()