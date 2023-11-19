import argparse
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt
from utils.monitor import load_results
from scipy.ndimage import gaussian_filter1d


X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: the input Array
    :param window: length of the rolling window
    :return: rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = (*array.strides, array.strides[-1])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: variable 1
    :param var_2: variable 2
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2 (such as np.mean)
    :return:  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1 :], function_on_var2


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var


def plot_trainingcurve_ub(
    path: str, x_axis: str,  task_name: str, num_timesteps: int = 25_000_000, figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param path: location containing the monitor files (must be in the same directory/folder)
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames, _ = load_results(path)
    for data_frame in data_frames:
        data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    
    plt.figure(task_name, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    
    interpolated_y_means = []
    for _, (x, y) in enumerate(xy_list):
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            interpolated_y_mean = np.interp(xy_list[0][0], x, y_mean)
            interpolated_y_means.append(interpolated_y_mean)

    combined_y_mean = np.mean(np.vstack(interpolated_y_means), axis=0)
    smoothed_y_mean = gaussian_filter1d(combined_y_mean, 600)
    std_devs = np.std(np.vstack(interpolated_y_means), axis=0)
    plt.plot(xy_list[0][0], smoothed_y_mean, label='Mean Across 5 Seeds (UB)')  
    plt.fill_between(xy_list[0][0], smoothed_y_mean - std_devs, smoothed_y_mean + std_devs, alpha=0.2, label='Std Across 5 Seeds (UB)')
    plt.xlim(min_x, max_x)
    plt.title(task_name)
    custom_ticks = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000]
    custom_labels = ['0', '5M', '10M', '15M', '20M', '25M']

    plt.xticks(custom_ticks, custom_labels)
    plt.xlabel("Timesteps (M)")
    plt.ylabel("Mean Returns")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_trainingcurve_subtask(
    path: str, task_name: str, x_axis: str = "timesteps", num_timesteps: int = 25_000_000, figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot the training data using csv files from ``Monitor`` wrapper.

    :param path: location containing the monitor files (must be in the same directory/folder)
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames, _ = load_results(path)
    for data_frame in data_frames:
        data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
    
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plt.figure(task_name, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    
    interpolated_y_means = []
    for _, (x, y) in enumerate(xy_list):
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            interpolated_y_mean = np.interp(xy_list[0][0], x, y_mean)
            interpolated_y_means.append(interpolated_y_mean)

    combined_y_mean = np.mean(np.vstack(interpolated_y_means), axis=0)
    smoothed_y_mean = gaussian_filter1d(combined_y_mean, 600)
    std_devs = np.std(np.vstack(interpolated_y_means), axis=0)
    plt.plot(xy_list[0][0], smoothed_y_mean, label='Mean Across All Models')
    plt.fill_between(xy_list[0][0], smoothed_y_mean - std_devs, smoothed_y_mean + std_devs, alpha=0.2, label='Std Across All Models')

    smoothed_curves = []
    for _, (x, y) in enumerate(xy_list):
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with a window of size EPISODES_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            interpolated_y_mean = np.interp(xy_list[0][0], x, y_mean)
            smoothed_y_mean = gaussian_filter1d(interpolated_y_mean, 600)
            
            # Store the smoothed curve in the list
            smoothed_curves.append(smoothed_y_mean)

    for curve in smoothed_curves:
        plt.plot(xy_list[0][0], curve)

    # order must be adjusted depending on the ranomd order the monitor files were drawn from, used to label individual models
    """plt.plot(xy_list[0][0], smoothed_curves[1], label='FB_fruits01')
    plt.plot(xy_list[0][0], smoothed_curves[3], label='FB_fruits12')
    plt.plot(xy_list[0][0], smoothed_curves[4], label='FB_fruits23')
    plt.plot(xy_list[0][0], smoothed_curves[2], label='FB_fruits34')
    plt.plot(xy_list[0][0], smoothed_curves[0], label='FB_fruits40')"""
    
    plt.xlim(min_x, max_x)
    plt.title(task_name)
    custom_ticks = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000]
    custom_labels = ['0', '5M', '10M', '15M', '20M', '25M']

    plt.xticks(custom_ticks, custom_labels)
    plt.xlabel("Timesteps (M)")
    plt.ylabel("Mean Returns")
    plt.legend(loc="lower right", fontsize="6")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help='name of the task the models were trained on (displayed in the plot)')
    parser.add_argument('--path', type=str, required=True, help='location of the monitor files (must be in same dir)')
    parser.add_argument('--type', type=str, required=True, choices=['ub', 'subtask'], help='specifies whether to create plot for upper bound or for models trained on a sub-task')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.type == 'ub':
        plot_trainingcurve_ub(path=args.path, num_timesteps=25_000_000, x_axis="timesteps", task_name=args.task_name)
    else:
        plot_trainingcurve_subtask(path=args.path, num_timesteps=25_000_000, x_axis="timesteps", task_name=args.task_name)


if __name__=="__main__":
    main()