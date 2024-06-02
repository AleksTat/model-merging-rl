# General information
- Code for merging models in RL, specifically models trained on Procgen environments
- The used model architecture and hyperparameters are crucial for the code to work, if one wants to modify any of the two, the code will need further modifications
- Always same settings for Procgen environments are used: easy, no background, and 64 instances in parallel
- Genereally, most Parser arguments are self explanatory but there is help information written for almost all of them to clarify their purpose

# Files
## train.py
- Used to train a model on a given Procgen environment
- Parser arguments should be self explanatory. "chp_timer" is the number of updates before a checkpoint of the model is created

## merge.py
- Merges two models, either by averaging their weights or by Git Re-Basin merging
- This means, two models need to be trained first and then this file needs to be used to merge them afterward, there is no file training two models first and then merging them afterward. Only the train_and_merge file does both, but for different merge settings, where merging is done during training and not afterward
- Code for Git Re-Basin is inside /utils/gitrebasin.py
- All Parser arguments should be self explanatory, except for gitrebasin_log, this is the filename gitrebasin will create for logging its permutation information (how much difference there was between the model weights)

## train_and_merge.py
- Contains functions for our federated approach (training and merging two models at the same time) and for our continual approach (training one model and merging it with corresponding checkpoints of a pre-trained model)
- Works similarly to train.py, but also imports the code from merge.py (extension of the train code to incorporate merging)
- Most difficult file to understand
- You have to give the Parser different arguments, depending on which of the functions you are using. The --do argument is used to specify which of the two you want to use
- Federated needs two envs, two model seeds etc., because two models are trained
- chp_merge needs only one env and model seed, but a path to the model of which its supposed to use the checkpoints
- The code for chp_merge is written so that it only works if you give it a path to a specific model that contains all relevant checkpoints with exactly the same name except for _{update_at_checkpoint_time}, which is the number that says after how many updates the checkpoint was created. This is exactly corresponding to the way train.py is saving checkpoints.
- With this chp_merge function, you can't merge end of training weights only with the currently training model_a, they always have to be corresponding checkpoints. However, the code can be easyly modified to change that, which I did for my experiments.

## evaluate.py
- Used to test models, spectate them or let a random agent play. Specify which of those with the --mode argument
- num_episodes=5000 was used for experiments; in one case =1000 was used
- seed=5 was used for all experiments
- Generates a monitor file in the case of mode test or random agent, and saves it to a given location. Doesn't create such a file for mode spectate

## plot_training.py
- Code can be used to create plots from monitor files. I used this in my thesis but not in the latest experiments, might require modifications to be used properly

## results.py
- Computes the results (like mean returns) after testing a model with evaluate.py and saving the test data in a csv file
- Writes the results to a new output file
- Can evaluate either episode returns or episode lengths.
- The monitor files to be evaluated have to be inside one folder together and the path to the folder needs to be given to the code

## test_merge.py
- Tests the implementations of the different merging procedures
- Currently not up to date and wasn't used lately!

## procgen_games
- Folder containing all procgen environments that were used as super-tasks and sub-tasks
- The individual game files should be added to a local Procgen installation (and registered, e.g. in cmake)

# Requirements
- All requirements can be found in the environment.yml file (you can use an alternative to conda). 
- Procgen is included in the environment.yml file, but commented out. We recommend a manual installation from the official Git repository (https://github.com/openai/procgen). Installing Procgen manually by first downloading the repository makes it easier to insert our modified Procgen files.
- **Cuda**: Cuda 11.7 was used for experiments with a RTX 3060, and the specific packages are included in the environment.yml file, but are commented out. This is because your machine might require a different cuda version or different packages. You can try uncommenting and installing the same ones, and if that doesn't work you need to find out which cuda version you can use, if any. You could simply install cuda with pip after setting up the conda environment by running `python -m pip install cuda-python` inside your terminal (after activating the conda environment). Cuda may also be installed by other dependencies without needing to uncomment it, so check this after conda environment creation.

# Install
Commands are specific to linux terminal.
1. Download this repository and open it inside the terminal.
2. Create conda environment: `conda env create -n mm-rl --file environment.yml
3. Activate conda environment: `conda activate mm-rl`
4. Install  cuda (*only if it is not already installed; check after creating conda environment*): `python -m pip install cuda-python`
5. Download Procgen repository fom https://github.com/openai/procgen
6. Open Procgen repository locally and insert our environment files from the procgen_games folder.
7. Register the new Procgen environments and install Procgen inside the crl-mm conda environment.
8. Open the crl-mm repository in your terminal, and run the files from there.
