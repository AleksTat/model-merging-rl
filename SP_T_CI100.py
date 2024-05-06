import subprocess

# Commands to run
commands = [
    ('python train.py --do just_train --total_steps 24_576_000 --env_a spdodge --seed_init_a 0 seed_env_a 0'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),
    
    ('python train.py --do just_train --total_steps 24_576_000 --env_a spdodge --seed_init_a 1 seed_env_a 1'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),

    ('python train.py --do just_train --total_steps 24_576_000 --env_a spdodge --seed_init_a 2 seed_env_a 2'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),

    ('python train.py --do just_train --total_steps 24_576_000 --env_a spshoot --seed_init_a 1 seed_env_a 2'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),

    ('python train.py --do just_train --total_steps 24_576_000 --env_a spshoot --seed_init_a 2 seed_env_a 0'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),

    ('python train.py --do just_train --total_steps 24_576_000 --env_a spshoot --seed_init_a 0 seed_env_a 1'
    '--monitor_dir ./monitor/SP/train/T/CI100/ '
    '--tensorboard_dir SP/T/CI100 '
    '--model_save_dir ./models/SP/T/CI100'),
]

# Loop to run each command exactly once
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
    else:
        print("Command executed successfully")