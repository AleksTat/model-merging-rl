from procgen import ProcgenEnv
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack, # if wanting to framestack environment observations, currently not used
)
from stable_baselines3.common.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', type=str, required=True, choices=['just_train', 'train_and_merge', 'checkpoint_merge'], 
                        help='used to specify which method to run')
    parser.add_argument('--env_a', type=str, required=True, 
                        help='specifies the environment of the model when training, and one of the models when also merging')
    parser.add_argument('--env_b', type=str, required=False, 
                        help='used when training and merging two models at the same time, specifies the environment of the second model')
    parser.add_argument('--seed_init_a', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_init_b', type=int, required=False, help='seed for model weight initialization')
    parser.add_argument('--seed_env_a', type=int, required=True, help='seed for environment level generation of the first model')
    parser.add_argument('--seed_env_b', type=int, required=False, help='seed for environment level generation of the second model')
    parser.add_argument('--model_path_a', type=str, required=True, help='save location for the first model after training')
    parser.add_argument('--model_path_b', type=str, required=False, help='save location for the second model after training')
    parser.add_argument('--monitor_path_a', type=str, required=True, help='save location for training data of the first model')
    parser.add_argument('--monitor_path_b', type=str, required=False, help='save location for training data of the second model')
    #parser.add_argument('--tensorboard_path', type=str, required=True, help='save location for tensorboard files')
    parser.add_argument('--total_steps', type=int, required=True, default=25_000_000, help='total timesteps for training')
    parser.add_argument('--merge_intervall', type=int, required=False, help='defines how many times parameters are updated before merging')
    return parser.parse_args()


def just_train(args):
    TOTAL_TIMESTEPS = args.total_steps

    print('Just training a model on', args.env)
    print('Seed for model initialization:', args.seed_init)
    print('Seed for level generation:', args.seed_env)
    print('Saving model after training to:', args.model_path)
    print('Monitor files for training are saved to:', args.monitor_path)
    print('Tensboard files are saved to', args.tb_path)
    
    set_random_seed(args.seed_init, using_cuda=True)
    
    tb_path= args.tb_path
    model_path = args.model_path
    monitor_path = args.monitor_path

    # always use same environment settings (could be changed in the future, for new experiments)
    venv = ProcgenEnv(num_envs=64, env_name=args.env, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=args.seed_env, use_backgrounds=False)
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv, filename=monitor_path)

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
                tensorboard_log=tb_path,
                verbose=1)
    
    #print(model.policy)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)


def train_and_merge(args):
    # 16384 is the n_steps (3) hyperparam times the environment instances (64), will use variables for this later but for now has to be this way
    TOTAL_UPDATES = args.total_steps / 16384 # calculates the total number of model updates during one training run
    MERGE_INTERVALL = args.merge_intervall
    USE_DIFFERENT_SEEDS = args.diff_seeds

    print('Training two models on', args.env)
    print('Seed for model initialization:', args.seed_init)
    print('Seed for level generation:', args.seed_env)
    print('Saving model after training to:', args.model_path)
    print('Monitor files for training are saved to:', args.monitor_path)
    print('Tensboard files are saved to', args.tb_path)
    
    set_random_seed(args.seed_init_a, using_cuda=True)
    
    
    model_path_a = './models/starpilotdodge1'
    model_path_b = './models/starpilotshoot2'
    monitor_path_a = './starpilotdodge1'
    monitor_path_b = './starpilotshoot2'

    venv_a = ProcgenEnv(num_envs=64, env_name=args.env_a, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=args.seed_env, use_backgrounds=False)
    venv_a = VecExtractDictObs(venv_a, 'rgb')
    venv_a = VecMonitor(venv=venv_a, filename=monitor_path_a)

    venv_b = ProcgenEnv(num_envs=64, env_name=args.env_b, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=args.seed_env, use_backgrounds=False)
    venv_b = VecExtractDictObs(venv_b, 'rgb')
    venv_b = VecMonitor(venv=venv_b, filename=monitor_path_b)

    model_a = PPO('CnnPolicy',
                venv_a, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log=None,
                verbose=1)
    
    if USE_DIFFERENT_SEEDS:
        set_random_seed(args.seed_init_b, using_cuda=True)

    model_b = PPO('CnnPolicy',
                venv_b, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log=None,
                verbose=1)
    
    for step in range(1, TOTAL_UPDATES+1):
        model_a.learn(total_timesteps=16384, tb_log_name='./todo', reset_num_timesteps=False)
        model_b.learn(total_timesteps=16384, tb_log_name='./todo', reset_num_timesteps=False)
        print('step:', step)

        # Merge model parameters every MERGE_INTERVALL iterations
        if step % MERGE_INTERVALL == 0:
            # Move parameters to cpu, then merge them, then move back to gpu and update models
            model_a_policy_cpu = {k: v.cpu() for k, v in model_a.policy.state_dict().items()}
            model_b_policy_cpu = {k: v.cpu() for k, v in model_b.policy.state_dict().items()}

            updated_params_cpu = gitrebasin(model_a_policy_cpu, model_b_policy_cpu, args.inter_param)
            updated_params = {k: v.cuda() for k, v in updated_params_cpu.items()}

            model_a.policy.load_state_dict(updated_params)
            model_b.policy.load_state_dict(updated_params)

    model_a.save(model_path_a)
    model_b.save(model_path_b)

def main():
    args = parse_args()
    
    print('env:', args.env)
    print('seed_init:', args.seed_init)
    print('seed_env:', args.seed_env)
    print('model_path:', args.model_path)
    print('monitor_path:', args.monitor_path)
    
    set_random_seed(args.seed_init, using_cuda=True)
    
    # use same schema for logs and model save directory
    # monitor_path = './monitor/train/'+args.env+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    # model_path = './models/'+str(args.env)+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    tb_path= './tensorboard/'+str(args.env)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    model_path = args.model_path
    monitor_path = args.monitor_path

    venv = ProcgenEnv(num_envs=64, env_name=args.env, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=args.seed_env, use_backgrounds=False)
    venv = VecExtractDictObs(venv, 'rgb')
    venv = VecMonitor(venv=venv, filename=monitor_path)

    model = PPO('CnnPolicy',
                venv, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
                tensorboard_log='./tensorboard',
                verbose=1)
    #print(model.policy)
    model.learn(total_timesteps=25_000_000, tb_log_name='./tensortest/pleasemodel', reset_num_timesteps=False)
    model.save(model_path)


if __name__ == '__main__':
    main()