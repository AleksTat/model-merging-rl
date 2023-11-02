from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt

from stable_baselines3.common.monitor import load_results
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
    print(x_axis)
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


def plot_curves(
    xy_list: List[Tuple[np.ndarray, np.ndarray]], xy_list2: List[Tuple[np.ndarray, np.ndarray]], 
    x_axis: str, title: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    interpolated_y_means = []
    interpolated_y_means2 = []
    
    for _, (x, y) in enumerate(xy_list):
        #plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            interpolated_y_mean = np.interp(xy_list[0][0], x, y_mean)
            interpolated_y_means.append(interpolated_y_mean)

    combined_y_mean = np.mean(np.vstack(interpolated_y_means), axis=0)
    smoothed_y_mean = gaussian_filter1d(combined_y_mean, 600)
    std_devs = np.std(np.vstack(interpolated_y_means), axis=0)
    plt.plot(xy_list[0][0], smoothed_y_mean, label='Upper Bound 1')  
    plt.fill_between(xy_list[0][0], smoothed_y_mean - std_devs, smoothed_y_mean + std_devs, alpha=0.2, label='Std of UB1')

    for _, (x, y) in enumerate(xy_list2):
        #plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            interpolated_y_mean2 = np.interp(xy_list2[0][0], x, y_mean)
            interpolated_y_means2.append(interpolated_y_mean2)

    combined_y_mean2 = np.mean(np.vstack(interpolated_y_means2), axis=0)
    smoothed_y_mean2 = gaussian_filter1d(combined_y_mean2, 600)
    std_devs2 = np.std(np.vstack(interpolated_y_means2), axis=0)
    plt.plot(xy_list2[0][0], smoothed_y_mean2, label='Upper Bound 2')  
    plt.fill_between(xy_list2[0][0], smoothed_y_mean2 - std_devs2, smoothed_y_mean2 + std_devs2, alpha=0.2, label='Std of UB2') 

    plt.xlim(min_x, max_x)
    plt.title(title)
    custom_ticks = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000, 25_000_000]
    custom_labels = ['0', '5M', '10M', '15M', '20M', '25M']

    plt.xticks(custom_ticks, custom_labels)
    plt.xlabel("Timesteps (M)")
    plt.ylabel("Mean Returns of the last 100 Episodes")
    plt.legend()
    plt.tight_layout()


def plot_results(
    dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, figsize: Tuple[int, int] = (8, 2)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    data_frames1 = data_frames[:len(data_frames)//2]
    data_frames2 = data_frames[len(data_frames)//2:]
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames1]
    xy_list2 = [ts2xy(data_frame, x_axis) for data_frame in data_frames2]
    plot_curves(xy_list, xy_list2, x_axis, task_name, figsize)

def plot_trainingcurve_ub(
    dirs: List[str], x_axis: str, num_timesteps: int = 25_000_000, task_name: str = "timesteps", figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    #data_frames = []
    for file in dirs:
        data_frames, _=load_results(file)
        """print(file)
        data_frame = load_results(file)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)"""
    for data_frame in data_frames:
        data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        print(len(data_frame))
    print(len(data_frames))
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    #plot_curves(xy_list, x_axis, task_name, figsize)
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

def plot_trainingcurve_subtasks(
    dirs: List[str], num_timesteps: Optional[int], task_name: str, x_axis: str = "timesteps", figsize: Tuple[int, int] = (4, 3)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    for file in dirs:
        data_frames, _=load_results(file)
        """print(file)
        data_frame = load_results(file)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)"""
    for data_frame in data_frames:
        data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        print(len(data_frame))
    print(len(data_frames))
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    print(xy_list[0][1])
    #plot_curves(xy_list, x_axis, task_name, figsize)
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

    # Plot the smoothed curves for individual (x, y) tuples
    for i, smoothed_curve in enumerate(smoothed_curves):
        if i == 0:
            label = f'SP_dodge22'
        if i == 1:
            label = f'SP_dodge00'
        if i == 2:
            label = f'SP_dodge44'
        if i == 3:
            label = f'SP_dodge11'
        if i == 4:
            label = f'SP_dodge33'
    plt.plot(xy_list[0][0], smoothed_curves[1], label='FB_fruits01')
    plt.plot(xy_list[0][0], smoothed_curves[3], label='FB_fruits12')
    plt.plot(xy_list[0][0], smoothed_curves[4], label='FB_fruits23')
    plt.plot(xy_list[0][0], smoothed_curves[2], label='FB_fruits34')
    plt.plot(xy_list[0][0], smoothed_curves[0], label='FB_fruits40')
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

if __name__=="__main__":
    plot_trainingcurve_ub(dirs=["/home/beksi/projects/thesis/monitor/train/SP/UB"], num_timesteps=25000000, x_axis="timesteps", task_name="Starpilot")