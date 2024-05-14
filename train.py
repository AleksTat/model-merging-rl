"""
This file is used to simply train a model on a given Procgen environment, given seeds and save paths for model and logs.
"""

import argparse
import os
import torch
from procgen import ProcgenEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack, # if wanting to framestack environment observations, currently not used
)
from stable_baselines3.common.utils import set_random_seed


def train(total_steps, env, seed_init, seed_env, model_save_dir, monitor_path, tb_path, chp_timer):
    TOTAL_UPDATES = round(total_steps / 16384)

    FILENAME = (f'{env}{seed_init}{seed_env}')
    model_save_folder = os.path.join(model_save_dir, FILENAME)
    os.makedirs(model_save_folder, exist_ok=True)

    # sets tensorboard log names and locations similar to above for the end model save location
    TB_LOG = os.path.join(tb_path, FILENAME)
    MONITOR = os.path.join(monitor_path, FILENAME)

    print('Training a model on', env)
    print('Seed for model initialization:', seed_init)
    print('Seed for level generation:', seed_env)
    print('Saving model after training to:', model_save_folder)
    print('Monitor files of training are saved to:', MONITOR)
    print('Tensboard files are saved to', TB_LOG)

    set_random_seed(seed_init, using_cuda=True)

    # always use same environment settings (could be changed in the future, for new experiments)
    venv = ProcgenEnv(num_envs=64, env_name=env, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed_env, use_backgrounds=False)
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv, filename=MONITOR)

    # always use the same hyperparameters (could be changed in the future, for new experiments)
    model = PPO('CnnPolicy',
                venv, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log="./tensorboard",
                verbose=1)
    
    #print(model.policy)
    for update in range(1, TOTAL_UPDATES+1):
        model.learn(total_timesteps=16384, tb_log_name=TB_LOG, reset_num_timesteps=False)

        if chp_timer != None:
            if update % chp_timer == 0:
                    checkpoint_name = FILENAME+f'_chp_{update}'
                    checkpoint_save_location = os.path.join(model_save_folder, checkpoint_name)
                    model.save(checkpoint_save_location)

    # folder has same name as model
    model_save_location = os.path.join(model_save_folder, FILENAME)
    model.save(model_save_location)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, 
                        help='specifies the environment in which the model is trained')
    parser.add_argument('--seed_init', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_env', type=int, required=True, help='seed for environment level generation')
    parser.add_argument('--model_save_dir', type=str, required=True, help='save location for the model after training')
    parser.add_argument('--monitor_dir', type=str, required=True, help='save location for training data')
    parser.add_argument('--tensorboard_dir', type=str, required=True, help='save location for tensorboard files')
    parser.add_argument('--total_steps', type=int, required=True, default=25_000_000, help='total timesteps for training')
    parser.add_argument('--chp_timer', type=int, default=500, help='Used to specify after how many updates checkpoints are created. Can be None')
    return parser.parse_args()


def main():
    args = parse_args()
    
    train(total_steps=args.total_steps, env=args.env, seed_init=args.seed_init, seed_env=args.seed_env, 
          model_save_dir=args.model_save_dir, monitor_path=args.monitor_dir, tb_path=args.tensorboard_dir, chp_timer=args.chp_timer)


if __name__ == '__main__':
    main()