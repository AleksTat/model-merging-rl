import subprocess

# Commands to run
commands = [
    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 50 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 0 --seed_env_a 0 '
    '--env_b fbtreats --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/FB/train/TM_OB/gitrebasin/diff_init '
    '--tensorboard_dir FB/TM_OB/gitrebasin/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/gitrebasin/diff_init'),
    
    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 75 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 1 --seed_env_a 1 '
    '--env_b fbtreats --seed_init_b 2 --seed_env_b 0 '
    '--monitor_dir ./monitor/FB/train/TM_OB/gitrebasin/diff_init '
    '--tensorboard_dir FB/TM_OB/gitrebasin/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/gitrebasin/diff_init'),

    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 100 --procedure gitrebasin --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 2 --seed_env_a 2 '
    '--env_b fbtreats --seed_init_b 0 --seed_env_b 1 '
    '--monitor_dir ./monitor/FB/train/TM_OB/gitrebasin/diff_init '
    '--tensorboard_dir FB/TM_OB/gitrebasin/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/gitrebasin/diff_init'),

    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 50 --procedure avg --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 0 --seed_env_a 0 '
    '--env_b fbtreats --seed_init_b 1 --seed_env_b 2 '
    '--monitor_dir ./monitor/FB/train/TM_OB/avg/diff_init '
    '--tensorboard_dir FB/TM_OB/avg/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/avg/diff_init'),
    
    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 75 --procedure avg --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 1 --seed_env_a 1 '
    '--env_b fbtreats --seed_init_b 2 --seed_env_b 0 '
    '--monitor_dir ./monitor/FB/train/TM_OB/avg/diff_init '
    '--tensorboard_dir FB/TM_OB/avg/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/avg/diff_init'),

    ('python train.py --do tm_special --total_steps 24_576_000 --merge_intervall 100 --procedure avg --inter_param 0.5 ' 
    '--env_a fbveggies --seed_init_a 2 --seed_env_a 2 '
    '--env_b fbtreats --seed_init_b 0 --seed_env_b 1 '
    '--monitor_dir ./monitor/FB/train/TM_OB/avg/diff_init '
    '--tensorboard_dir FB/TM_OB/avg/diff_init '
    '--model_save_dir ./models/FB/merged/TM_OB/avg/diff_init')
]

# Loop to run each command exactly once
for command in commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
    else:
        print("Command executed successfully")