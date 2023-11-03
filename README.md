## Files
- train.py: Used to train the models
	- I used all these hyperparameters for every training run
- merge.py: Used to merge models, either use gitrebasin or weight averaging
	- Code for gitrebasin is inside /utils/gitrebasin.py
- evaluate.py: Used to test models, spectate them or let a random agent play, specify which of those with the --mode argument
	- I used num_episodes=5000 for my expirments, in on case I used =1000
	- I used seed=5 always
- plots_and_tableentries: Contains the files I used to plot training data (plot_training.py) and compute the mean return values etc. that I have in my tables (results_table.py)
	- Very manual usage, changed which function to run by hand each time because I did this at the end and there was little time
        - Requires a slight change in stable-baselines3 package code to work  	
	- Uses most of stable-baselines3 stuff
- procgen_games: Contains all procgen game files I used for the super-tasks and sub-tasks with my modifications
- test_merge.py: Used this to confirm that my merging functions are functioning correctly, it is a bit messy right now and a lot of things are commented out. I also did other things to verify correct merging.

## Models
- Currently includes all my super-task and sub-task models, and all averaged models
- Gitrebasin models will follow, they are many

## Requirements
- gym 0.21.0
- gym3 0.3.3
- numpy 1.21.6
- nvidia-cublas-cu11 11.10.3.66 
- nvidia-cuda-nvrtc-cu11 11.7.99
- nvidia-cuda-runtime-cu11 11.7.99
- nvidia-cudnn-cu11 8.5.0.96              
- procgen 0.10.7
- python 3.7.16
- scipy 1.7.3
- stable baselines3 1.8.0
- torch 1.13.1
