from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt

from stable_baselines3.common.monitor import load_results
from scipy.ndimage import gaussian_filter1d
    

def plot_results(
    dir: str) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = load_results(dir)
    episode_returns = data_frames[0].r.values
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)
    """plt.figure(figsize=(8, 6))
    plt.hist(episode_returns, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Episode Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()"""
    print("mean:", mean_return)
    print("std:",std_return)
    print("min_return:", min_return)
    print("max_return:", max_return)

def save_results_to_file(dir: str, output_file: str) -> None:
    """
    Calculate and save the results to a file using csv files from ``Monitor`` wrapper.
    
    :param dir: the directory containing monitor files
    :param output_file: the file path to save the results
    """
    data_frames, file_names = load_results(dir)
    
    # List to store mean returns for each file
    mean_returns = []
    std_returns = []
    overall_returns = np.array([])

    for data_frame in data_frames:
        episode_returns = data_frame.r.values
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        overall_returns=np.concatenate((overall_returns, data_frame.r.values))
        # Append the mean return to the list
        mean_returns.append(mean_return)
        std_returns.append(std_return)

    # Calculate the overall mean and standard deviation across all files
    print(len(overall_returns))
    overall_mean = np.mean(overall_returns)
    overall_std = np.std(overall_returns)
    
    # Save the results to a file
    with open(output_file, "w") as file:
        file.write("File, Mean, Std\n")
        for file_name, data_frame, mean_return, std_return in zip(file_names,data_frames, mean_returns, std_returns):
            file.write(f"{file_name}, {mean_return}, {std_return}\n")
        file.write(f"Overall, {overall_mean}, {overall_std}\n")

def lengths(dir: str, output_file: str) -> None:
    """
    Calculate and save the results to a file using csv files from ``Monitor`` wrapper.
    
    :param dir: the directory containing monitor files
    :param output_file: the file path to save the results
    """
    data_frames, file_names = load_results(dir)
    
    # List to store mean returns for each file
    mean_lengths = []
    std_lengths = []
    overall_returns = np.array([])

    for data_frame in data_frames:
        episode_lengths = data_frame.l.values
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        overall_returns=np.concatenate((overall_returns, data_frame.l.values))
        # Append the mean return to the list
        mean_lengths.append(mean_length)
        std_lengths.append(std_length)

    # Calculate the overall mean and standard deviation across all files
    print(len(overall_returns))
    overall_mean = np.mean(overall_returns)
    overall_std = np.std(overall_returns)
    
    # Save the results to a file
    with open(output_file, "w") as file:
        file.write("File, mean, std")
        for file_name, data_frame, mean_length, std_length in zip(file_names,data_frames, mean_lengths, std_lengths):
            file.write(f"{file_name}, {mean_length}, {std_length}\n")
        file.write(f"Overall, {overall_mean}, {overall_std}\n")

if __name__ == "__main__":
    lengths("./", 
            "./lengths_ra")