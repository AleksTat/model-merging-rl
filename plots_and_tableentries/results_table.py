from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt

from stable_baselines3.common.monitor import load_results


def save_returns_to_file(dir: str, output_file: str) -> None:
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

def save_lengths_to_file(dir: str, output_file: str) -> None:
    """
    Calculate and save the results to a file using csv files from ``Monitor`` wrapper.
    
    :param dir: the directory containing monitor files
    :param output_file: the file path to save the results
    """
    data_frames, file_names = load_results(dir)
    
    # List to store mean lengths for each file
    mean_lengths = []
    std_lengths = []
    overall_lengths = np.array([])

    for data_frame in data_frames:
        episode_lengths = data_frame.l.values
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        overall_lengths=np.concatenate((overall_lengths, data_frame.l.values))
        # Append the mean return to the list
        mean_lengths.append(mean_length)
        std_lengths.append(std_length)

    # Calculate the overall mean and standard deviation across all files
    print(len(overall_lengths))
    overall_mean = np.mean(overall_lengths)
    overall_std = np.std(overall_lengths)
    
    # Save the results to a file
    with open(output_file, "w") as file:
        file.write("File, mean, std")
        for file_name, data_frame, mean_length, std_length in zip(file_names,data_frames, mean_lengths, std_lengths):
            file.write(f"{file_name}, {mean_length}, {std_length}\n")
        file.write(f"Overall, {overall_mean}, {overall_std}\n")

if __name__ == "__main__":
    save_lengths_to_file("./", 
            "./lengths_ra")