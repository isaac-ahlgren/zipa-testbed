import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

from utils import parse_eval_directory_time_stamps, make_plot_dir, extract_from_contents, get_num_events_list

def event_hist_plot(devices, contents, param1, param2, param1_range, param2_range, savefigs=True, fig_dir=None, file_name=None, grid_resolution=100):
    for device in devices:
        device_id = device + "_time_stamps"

        device_event_list = extract_from_contents(contents, device_id)

        event_num_list = get_num_events_list(device_event_list)

        params = extract_from_contents(contents, "params")
        param1_list = extract_from_contents(params, param1)
        param2_list = extract_from_contents(params, param2)

        grid_x, grid_y = np.linspace(min(param1_list), max(param1_list), grid_resolution), np.linspace(min(param2_list), max(param2_list), grid_resolution)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        grid_z = griddata((param1_list, param2_list), event_num_list, (grid_x, grid_y), method='cubic')

        fig, ax = plt.subplots()
        c = ax.pcolormesh(grid_z)
        ax.set_title(f'Heat Map of Events {device}')
        ax.axis([param1_range[0], param1_range[1], param2_range[0], param2_range[1]])
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)

        if savefig:
            plt.savefig(fig_dir + "/" + file_name + "_" + device + ".pdf")
            plt.clf()
            fig_data_name = fig_dir + "/" + file_name + ".csv"
            df = pd.DataFrame({"x_axis": param1_list, "y_axis": param2_list, "z_axis": event_num_list})
            df.to_csv(fig_data_name)
        else:
            plt.show()

def plot_perceptio(savefigs=True):
    PERCEPTIO_DATA_DIRECTORY = "../perceptio/perceptio_data/perceptio_real_fuzz/perceptio_real_event_fuzz"
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
                    (100, 2 * 133300224), (100, 2 * 133300224), 
                    savefigs=savefigs, fig_dir=FIG_DIR_NAME_STUB, file_name=PERCEPTIO_REAL_FUZZING_STUB)

def plot_iotcupid(savefigs=True):
    IOTCUPID_DATA_DIRECTORY = "../iotcupid/iotcupid_data/iotcupid_real_fuzz/iotcupid_real_event_fuzz"
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
                    savefigs=savefigs, fig_dir=FIG_DIR_NAME_STUB, file_name=IOTCUPID_REAL_FUZZING_STUB)

if __name__ == "__main__":
    plot_perceptio()
    plot_iotcupid()

    

