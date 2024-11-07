import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import pandas as pd

from utils import parse_eval_directory_time_stamps, make_plot_dir, extract_from_contents, get_num_events_list

def event_hist_plot(devices, contents, param1, param2, param1_range, param2_range, savefigs=True, fig_dir=None, file_name=None, grid_size=100):
    for device in devices:
        device_id = device + "_time_stamps"

        device_event_list = extract_from_contents(contents, device_id)

        event_num_list = get_num_events_list(device_event_list)

        params = extract_from_contents(contents, "params")

        param1_list = extract_from_contents(params, param1)
        param2_list = extract_from_contents(params, param2)
 
        heatmap, xedges, yedges = np.histogram2d(param1_list, param2_list, bins=grid_size, weights=event_num_list)

        fig, ax = plt.subplots()
        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest', extent=[min(xedges), max(xedges), min(yedges), max(yedges)])        
        ax.set_title(f'Heat Map of Events {device}')
        ax.axis([param1_range[0], param1_range[1], param2_range[0], param2_range[1]])
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)

        if savefigs:
            plt.savefig(fig_dir + "/" + file_name + "_" + device + ".pdf")
            plt.clf()
            fig_data_name = fig_dir + "/" + file_name + "_" + device + ".csv"
            df = pd.DataFrame({"x_axis": param1_list, "y_axis": param2_list, "z_axis": event_num_list})
            df.to_csv(fig_data_name)
            plt.close()
        else:
            plt.show()

def event_hist_3d_plot(devices, contents, param1, param2, param3, savefigs=True, fig_dir=None, file_name=None):
    for device in devices:
        device_id = device + "_time_stamps"

        device_event_list = extract_from_contents(contents, device_id)

        event_num_list = get_num_events_list(device_event_list)

        params = extract_from_contents(contents, "params")

        param1_list = extract_from_contents(params, param1)
        param2_list = extract_from_contents(params, param2)
        param3_list = extract_from_contents(params, param2)
 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D scatter plot, with color representing "heat" (w_coords)
        scatter = ax.scatter(param1_list, param2_list, param3_list, c=event_num_list, cmap='hot')       
        ax.set_title(f'Heat Map of Events {device}')
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_xlabel(param3)

        if savefigs:
            plt.savefig(fig_dir + "/" + file_name + "_" + device + ".pdf")
            plt.clf()
            fig_data_name = fig_dir + "/" + file_name + "_" + device + ".csv"
            df = pd.DataFrame({"x_axis": param1_list, "y_axis": param2_list, "z_axis": param3_list, "intensity": event_num_list})
            df.to_csv(fig_data_name)
            plt.close()
        else:
            plt.show()


def plot_perceptio(savefigs=True):
    PERCEPTIO_DATA_DIRECTORY = "../perceptio/perceptio_data/perceptio_real_fuzz"
    PERCEPTIO_REAL_FUZZING_STUB = "perceptio_real_event_fuzz"
    FIG_DIR_NAME_STUB = "perceptio_real_event_fuzz_event_histogram"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory_time_stamps(
        f"{PERCEPTIO_DATA_DIRECTORY}",
        f"{PERCEPTIO_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    event_hist_plot(DEVICES, contents, "top_th", "bottom_th", 
                    (100, 50000000), (100, 50000000), 
                    savefigs=savefigs, fig_dir=fig_dir, file_name=PERCEPTIO_REAL_FUZZING_STUB)

def plot_iotcupid(savefigs=True):
    IOTCUPID_DATA_DIRECTORY = "../iotcupid/iotcupid_data/iotcupid_real_fuzz"
    IOTCUPID_REAL_FUZZING_STUB = "iotcupid_real_event_fuzz"
    FIG_DIR_NAME_STUB = "iotcupid_real_event_fuzz_event_histogram"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory_time_stamps(
        f"{IOTCUPID_DATA_DIRECTORY}",
        f"{IOTCUPID_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    event_hist_plot(DEVICES, contents, "top_th", "bottom_th", 
                    (0.00001, 0.1), (0.00001, 0.1), 
                    savefigs=savefigs, fig_dir=fig_dir, file_name=IOTCUPID_REAL_FUZZING_STUB)

def plot_fastzip(savefigs=True):
    FASTZIP_DATA_DIRECTORY = "../fastzip/fastzip_data/fastzip_real_fuzz"
    FASTZIP_REAL_FUZZING_STUB = "fastzip_event_real_fuzz"
    FIG_DIR_NAME_STUB = "fastzip_real_event_fuzz_event_histogram"

    DEVICES = [
        "10.0.0.238",
        "10.0.0.228",
        "10.0.0.231",
        "10.0.0.232",
        "10.0.0.233",
        "10.0.0.236",
        "10.0.0.227",
        "10.0.0.229",
        "10.0.0.235",
        "10.0.0.237",
        "10.0.0.234",
        "10.0.0.239",
    ]

    contents = parse_eval_directory_time_stamps(
        f"{FASTZIP_DATA_DIRECTORY}",
        f"{FASTZIP_REAL_FUZZING_STUB}",
    )

    if savefigs:
        fig_dir = make_plot_dir(FIG_DIR_NAME_STUB)
    else:
        fig_dir = None

    event_hist_3d_plot(DEVICES, contents, "power_th", "snr_th", "peak_th", savefigs=savefigs, fig_dir=fig_dir, file_name=FASTZIP_REAL_FUZZING_STUB)

if __name__ == "__main__":
    plot_perceptio()
    plot_iotcupid()
    plot_fastzip()

    

