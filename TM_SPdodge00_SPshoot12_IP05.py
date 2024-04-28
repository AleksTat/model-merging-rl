import subprocess

# Commands to run
commands = [
    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 10 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/gitrebasin/diff_init '
    '--tensorboard_dir SP/TM/gitrebasin/diff_init '
    '--model_save_dir ./models/SP/merged/TM/gitrebasin/diff_init'),
    
    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 50 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/gitrebasin/diff_init '
    '--tensorboard_dir SP/TM/gitrebasin/diff_init '
    '--model_save_dir ./models/SP/merged/TM/gitrebasin/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 100 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/gitrebasin/diff_init '
    '--tensorboard_dir SP/TM/gitrebasin/diff_init '
    '--model_save_dir ./models/SP/merged/TM/gitrebasin/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 500 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/gitrebasin/diff_init '
    '--tensorboard_dir SP/TM/gitrebasin/diff_init '
    '--model_save_dir ./models/SP/merged/TM/gitrebasin/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 10 --procedure avg --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/avg/diff_init '
    '--tensorboard_dir SP/TM/avg/diff_init '
    '--model_save_dir ./models/SP/merged/TM/avg/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 50 --procedure avg --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/avg/diff_init '
    '--tensorboard_dir SP/TM/avg/diff_init '
    '--model_save_dir ./models/SP/merged/TM/avg/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 100 --procedure avg --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/avg/diff_init '
    '--tensorboard_dir SP/TM/avg/diff_init '
    '--model_save_dir ./models/SP/merged/TM/avg/diff_init'),

    ('python train.py --do train_and_merge --total_steps 24_576_000 --merge_intervall 500 --procedure avg --inter_param 0.5 ' 
    '--env_a spdodge --seed_init_a 0 --seed_env_a 0 '
    '--env_b spshoot --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/SP/train/TM/avg/diff_init '
    '--tensorboard_dir SP/TM/avg/diff_init '
    '--model_save_dir ./models/SP/merged/TM/avg/diff_init')
]

# Loop to run each command exactly once
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
    else:
        print("Command executed successfully")