# General information
- Code for bachelor thesis "Continual Reinforcement Learning by Merging Models"
- Use to reproduce the results or to conduct further experiments

# Files
## train.py
- Trains models
- Hyperparameters stayed the same for every training run in the thesis
## merge.py
- Merges two models, either by averaging their weights or by Git Re-Basin merging
- Code for Git Re-Basin is inside /utils/gitrebasin.py
## evaluate.py
- Used to test models, spectate them or let a random agent play. Specify which of those with the --mode argument
- num_episodes=5000 was used for experiments; in one case =1000 was used
- seed=5 was used for all experiments
## plot_training.py
- Creates training plots of both UB and sub-task models
## results.py
- Computes the results (like mean returns) after testing a model with evaluate.py and saving the test data in a csv file
- Writes the results in a new output file
## test_merge.py
- Tests the implementations of the different merging procedures
- Currently not up to date!
## procgen_games
- Folder containing all procgen environments that were used as super-tasks and sub-tasks
- The individual game files should be added to a local procgen installation (and registered, e.g. in cmake)
- test_merge.py: Used this to confirm that my merging functions are functioning correctly, it is a bit messy right now and a lot of things are commented out. I also did other things to verify correct merging.

# Models
## Starpilot
- All super-task models (UB), sub-task models and merged models for which results are reported in the thesis can be found inside the models_starpilot folder
## Fruitbot
- All super-task models (UB), sub-task models and merged models for which results are reported in the thesis can be found inside the models_fruitbot folder

# Requirements
- All requirements can be found in the environment.yml file (you can use an alternative to conda). 
- Procgen is included in the environment.yml file, but commented out. We recommend a manual installation from the official Git repository (https://github.com/openai/procgen). Installing Procgen manually by first downloading the repository makes it easier to insert our modified Procgen files.
- **Cuda**: Cuda 11.7 was used for experiments with a RTX 3060, and the specific packages are included in the environment.yml file, but are commented out. This is because your machine might require a different cuda version or different packages. You can try uncommenting and installing the same ones, and if that doesn't work you need to find out which cuda version you can use, if any. You could simply install cuda with pip after setting up the conda environment by running `python -m pip install cuda-python` inside your terminal (after activating the conda environment). Cuda may also be installed by other dependencies without needing to uncomment it, so check this after conda environment creation.

# Install
Commands are specific to linux terminal.
1. Download this repository and open it inside the terminal.
2. Create conda environment: `conda env create -n crl-mm --file environment.yml
3. Activate conda environment: `conda activate crl-mm`
4. Install  cuda (*only if it is not already installed; check after creating conda environment*): `python -m pip install cuda-python`
5. Download Procgen repository fom https://github.com/openai/procgen
6. Open Procgen repository locally and insert our environment files from the procgen_games folder.
7. Register the new Procgen environments and install Procgen inside the crl-mm conda environment.
8. Open the crl-mm repository in your terminal, and run the files from there.

# Reproduce
The following lists every single step to (hopefully) reproduce our experiment results when merging models trained on the Starpilot sub-tasks using Git Re-Basin with alpha=0.5.
1. `python train.py --env starpilot_dodge --seed_init 0 --seed_env 0 --model_path ./reproduce_models_starpilot/SP_dodge/SP_dodge00 --monitor_path
