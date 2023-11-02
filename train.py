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
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--seed_init', type=int, required=True)
    parser.add_argument('--seed_env', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--monitor_path', type=str, required=True)
    return parser.parse_args()

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
    tb_path= './tensorboard/'+str(args.env)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
   # model_path = './models/'+str(args.env)+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
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
                tensorboard_log=tb_path,
                verbose=1)
    #print(model.policy)
    model.learn(total_timesteps=25_000_000)
    model.save(model_path)

if __name__ == '__main__':
    main()