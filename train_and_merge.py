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
from merge import gitrebasin, weight_averaging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, required=True, choices=['federated', 'chp_merge'], 
                        help='used to specify which method to run. can choose from federated (training and merging 2 models) '
                             'and chp_merge (training one model and merging it with checkpoints of other model that was trained previously)')
    parser.add_argument('--env_a', type=str, required=True, 
                        help='specifies the environment of the model when training, and one of the models when also merging')
    parser.add_argument('--env_b', type=str, required=False, 
                        help='used when training and merging two models at the same time, specifies the environment of the second model')
    parser.add_argument('--seed_init_a', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_init_b', type=int, required=False, help='seed for model weight initialization')
    parser.add_argument('--seed_env_a', type=int, required=True, help='seed for environment level generation of the first model')
    parser.add_argument('--seed_env_b', type=int, required=False, help='seed for environment level generation of the second model')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='save location for the model after training or merging (always only saving one model)')
    parser.add_argument('--monitor_dir', type=str, required=True, help='save location for training data of the first model')
    parser.add_argument('--tb_dir', type=str, required=True, help='save location for tensorboard files')
    parser.add_argument('--total_steps', type=int, required=True, default=25_000_000, help='total timesteps for training')
    parser.add_argument('--merge_intervall', type=int, required=False, help='defines how many times parameters are updated before merging')
    parser.add_argument('--procedure', type=str, required=False, choices=['avg', 'gitrebasin'],
                        help='which merging procedure to use. possible only (avg, gitrebasin)')
    parser.add_argument('--inter_param', type=float, required=False, help='interpolation parameter for the merging')
    parser.add_argument('--model_b_path', type=str, required=False, 
                        help='path to a folder that contains model and checkpoints that is used in checkpoint merging')
    parser.add_argument('--model_b_name', type=str, required=False, 
                        help='giving name of model_b for checkpoint merging so that appropriate filename can be created for merged model later')
    return parser.parse_args()

# train and merge two models simultaneously
def federated(total_steps, merge_intervall, procedure, inter_param, env_a, env_b, seed_init_a, seed_init_b, 
                    seed_env_a, seed_env_b, monitor_dir, tb_dir, model_save_dir):
    # 16384 is the n_steps (256) hyperparam times the environment instances (64), 
    # will use variables for this later but for now has to be this way
    TOTAL_UPDATES = round(total_steps / 16384)
    # calculates the total number of model updates during one training run

    # model name after saving, also used for checkpoints
    MODEL_SAVE_FILENAME = (f'TM_MI{merge_intervall}_{procedure}_IP{str(inter_param).replace(".","")}_{env_a}{seed_init_a}'
                           f'{seed_env_a}_{env_b}{seed_init_b}{seed_env_b}')
    LOG_FILE_NAME_A = (f'{env_a}{seed_init_a}{seed_env_a}_{env_b}{seed_init_b}{seed_env_b}_TM_MI{merge_intervall}_{procedure}_'
                       f'IP{str(inter_param).replace(".","")}_{env_a}{seed_init_a}{seed_env_a}')
    LOG_FILE_NAME_B = (f'{env_a}{seed_init_a}{seed_env_a}_{env_b}{seed_init_b}{seed_env_b}_TM_MI{merge_intervall}_{procedure}_'
                       f'IP{str(inter_param).replace(".","")}_{env_b}{seed_init_b}{seed_env_b}')
    # create a folder inside the save directory with the same name as the model. there the model and checkpoints are saved
    model_save_folder = os.path.join(model_save_dir, MODEL_SAVE_FILENAME)
    os.makedirs(model_save_folder, exist_ok=True)

    # sets tensorboard log names and locations similar to above for the end model save location
    TB_LOG_A = os.path.join(tb_dir, LOG_FILE_NAME_A)
    TB_LOG_B = os.path.join(tb_dir, LOG_FILE_NAME_B)
    
    # sets monitor file names and locations similar to above for the end model save location
    MONITOR_A = os.path.join(monitor_dir, LOG_FILE_NAME_A)
    MONITOR_B = os.path.join(monitor_dir, LOG_FILE_NAME_B)
    
    if procedure == 'gitrebasin':
        permutation_diff_log = os.path.join(model_save_folder, MODEL_SAVE_FILENAME+'.txt') 
    
    print('total number of updates:', TOTAL_UPDATES)
    print('Training model a on', env_a)
    print('Training model b on', env_b)
    print('Total timesteps:', total_steps)
    print('Merging models every %s iterations' % merge_intervall)
    print('Merging procedure:', procedure)
    print('Interpolation parameter:', inter_param)
    print('Seeds for model a (init, env):', seed_init_a, seed_env_a)
    print('Seeds for model b (init, env):', seed_init_b, seed_env_b)
    print('Saving resulting model and checkpoints to:', model_save_folder)
    print('Monitor files of model a are saved to:', MONITOR_A)
    print('Monitor files of model b are saved to:', MONITOR_B)
    print('Tensboard files of model a are saved to', TB_LOG_A)
    print('Tensboard files of model b are saved to', TB_LOG_B)
    
    set_random_seed(seed_init_a, using_cuda=True)
    venv_a = ProcgenEnv(num_envs=64, env_name=env_a, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed_env_a, use_backgrounds=False)
    venv_a = VecExtractDictObs(venv_a, 'rgb')
    venv_a = VecMonitor(venv=venv_a, filename=MONITOR_A)
    model_a = PPO('CnnPolicy',
                venv_a, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log='./tensorboard',
                verbose=1)
    
    set_random_seed(seed_init_b, using_cuda=True)
    venv_b = ProcgenEnv(num_envs=64, env_name=env_b, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed_env_b, use_backgrounds=False)
    venv_b = VecExtractDictObs(venv_b, 'rgb')
    venv_b = VecMonitor(venv=venv_b, filename=MONITOR_B)
    model_b = PPO('CnnPolicy',
                venv_b, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log='./tensorboard',
                verbose=1)
    
    for update in range(1, TOTAL_UPDATES+1):
        model_a.learn(total_timesteps=16384, tb_log_name=TB_LOG_A, reset_num_timesteps=False)
        model_b.learn(total_timesteps=16384, tb_log_name=TB_LOG_B, reset_num_timesteps=False)
        print('update iteration', update)

        # Merge model parameters every merge_intervall iterations (optimizers are never merged!)
        if update % merge_intervall == 0:
            if procedure == 'gitrebasin':
                # Move parameters to cpu, then merge them, then move back to gpu and update models
                params_model_a_cpu = {k: v.cpu() for k, v in model_a.policy.state_dict().items()}
                params_model_b_cpu = {k: v.cpu() for k, v in model_b.policy.state_dict().items()}

                updated_params_cpu = gitrebasin(params_model_a_cpu, params_model_b_cpu, inter_param, output_file=permutation_diff_log)
                updated_params = {k: v.cuda() for k, v in updated_params_cpu.items()}

            elif procedure == 'avg':
                updated_params = weight_averaging(model_a.policy.state_dict(), model_b.policy.state_dict(), inter_param=inter_param)

            model_a.policy.load_state_dict(updated_params)
            model_b.policy.load_state_dict(updated_params)

            # Save model parameter checkpoints every 500 updates (note that this does not save optimizers!)
            if update % 500 == 0 and update != 1500:
                checkpoint_path = MODEL_SAVE_FILENAME+f'_chp_{update}.pt'
                checkpoint_save_location = os.path.join(model_save_folder, checkpoint_path)
                torch.save(updated_params, checkpoint_save_location)

    # saved model and the folder in which it is saved have the same names
    model_save_location = os.path.join(model_save_folder, MODEL_SAVE_FILENAME)
    
    # depending on whether or not total_steps is a multiple of 16384, training does continue a little bit after the last merge or it doesn't
    model_b.save(model_save_location)


def chp_merge(total_steps, merge_intervall, procedure, inter_param, env_a, seed_init_a, seed_env_a, 
              model_b_path, model_b_name, monitor_dir, tb_dir, model_save_dir):
    # 16384 is the n_steps (256) hyperparam times the environment instances (64), 
    # will use variables for this later but for now has to be this way
    TOTAL_UPDATES = round(total_steps / 16384)
    # calculates the total number of model updates during one training run

    # model name after saving, also used for checkpoints
    MODEL_SAVE_FILENAME = (f'{env_a}{seed_init_a}{seed_env_a}_{model_b_name}_CM_MI{merge_intervall}_{procedure}_'
                           f'IP{str(inter_param).replace(".","")}_{env_a}{seed_init_a}{seed_env_a}')
    # create a folder inside the save directory with the same name as the model. there the model and checkpoints are saved
    model_save_folder = os.path.join(model_save_dir, MODEL_SAVE_FILENAME)
    os.makedirs(model_save_folder, exist_ok=True)

    # sets tensorboard log names and locations similar to above for the end model save location
    TB_LOG_A = os.path.join(tb_dir, MODEL_SAVE_FILENAME)
    
    # sets monitor file names and locations similar to above for the end model save location
    MONITOR_A = os.path.join(monitor_dir, MODEL_SAVE_FILENAME)
    
    if procedure == 'gitrebasin':
        permutation_diff_log = os.path.join(model_save_folder, MODEL_SAVE_FILENAME+'.txt') 
    
    print('total number of updates:', TOTAL_UPDATES)
    print('Training model a on', env_a)
    print('Total timesteps:', total_steps)
    print('Merging models every %s iterations' % merge_intervall)
    print('Merging procedure:', procedure)
    print('Interpolation parameter:', inter_param)
    print('Seeds for model a (init, env):', seed_init_a, seed_env_a)
    print('Saving resulting model and checkpoints to:', model_save_folder)
    print('Monitor files of model a are saved to:', MONITOR_A)
    print('Tensboard files of model a are saved to', TB_LOG_A)
    
    set_random_seed(seed_init_a, using_cuda=True)
    venv_a = ProcgenEnv(num_envs=64, env_name=env_a, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed_env_a, use_backgrounds=False)
    venv_a = VecExtractDictObs(venv_a, 'rgb')
    venv_a = VecMonitor(venv=venv_a, filename=MONITOR_A)
    model_a = PPO('CnnPolicy',
                venv_a, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log='./tensorboard',
                verbose=1)

    for update in range(1, TOTAL_UPDATES+1):
        model_a.learn(total_timesteps=16384, tb_log_name=TB_LOG_A, reset_num_timesteps=False)
        print('update iteration', update)

        # Merge model parameters every merge_intervall iterations (optimizers are never merged!)
        if update % merge_intervall == 0:
            # this assumes model b is saved with all its checkpoints in one folder and we merge with corresponding checkpoint to current update
            # can be modified to only merge with end model weights of model_b
            model_b_file = model_b_path + f'{update}.zip'
            print('model_b_file:', model_b_file)
            # env is not important here, so just using model_a's
            model_b = PPO.load(model_b_file, venv_a)
            if procedure == 'gitrebasin':
                # Move parameters to cpu, then merge them, then move back to gpu and update models
                params_model_a_cpu = {k: v.cpu() for k, v in model_a.policy.state_dict().items()}
                params_model_b_cpu = {k: v.cpu() for k, v in model_b.policy.state_dict().items()}

                updated_params_cpu = gitrebasin(params_model_a_cpu, params_model_b_cpu, inter_param, output_file=permutation_diff_log)
                updated_params = {k: v.cuda() for k, v in updated_params_cpu.items()}

            elif procedure == 'avg':
                updated_params = weight_averaging(model_a.policy.state_dict(), model_b.policy.state_dict(), inter_param=inter_param)

            model_a.policy.load_state_dict(updated_params)

            # Save model parameter checkpoints every 500 updates
            if update % 500 == 0 and update != 1500:
                checkpoint_path = MODEL_SAVE_FILENAME+f'_chp_{update}.pt'
                checkpoint_save_location = os.path.join(model_save_folder, checkpoint_path)
                model_a.save(checkpoint_save_location)

    # saved model and the folder in which it is saved have the same names
    model_save_location = os.path.join(model_save_folder, MODEL_SAVE_FILENAME)
    # depending on whether or not total_steps is a multiple of 16384, training does continue a little bit after the last merge or it doesn't
    model_a.save(model_save_location)


def main():
    args = parse_args()
    
    if args.do == 'federated':
        federated(total_steps=args.total_steps, merge_intervall=args.merge_intervall, procedure=args.procedure,  
                        inter_param=args.inter_param, env_a=args.env_a, env_b=args.env_b, seed_init_a=args.seed_init_a,  
                        seed_init_b=args.seed_init_b, seed_env_a=args.seed_env_a, seed_env_b=args.seed_env_b, 
                        monitor_dir=args.monitor_dir, tb_dir=args.tb_dir, model_save_dir=args.model_save_dir)
        
    elif args.do == 'chp_merge': 
        chp_merge(total_steps=args.total_steps, merge_intervall=args.merge_intervall, procedure=args.procedure,  
                    inter_param=args.inter_param, env_a=args.env_a, seed_init_a=args.seed_init_a, seed_env_a=args.seed_env_a, 
                    monitor_dir=args.monitor_dir, tb_dir=args.tb_dir, model_save_dir=args.model_save_dir,
                    model_b_path=args.model_b_path, model_b_name=args.model_b_name)
    return 1


if __name__ == '__main__':
    main()