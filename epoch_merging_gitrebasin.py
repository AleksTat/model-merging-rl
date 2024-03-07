from procgen import ProcgenEnv
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack, # if wanting to framestack environment observations, currently not used
)
from stable_baselines3.common.utils import set_random_seed
from merge import gitrebasin

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_a', type=str, required=True)
    parser.add_argument('--env_b', type=str, required=True)
    parser.add_argument('--seed_init', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_env', type=int, required=True, help='seed for environment level generation')
    #parser.add_argument('--model_path', type=str, required=True, help='save location for model after training')
    #parser.add_argument('--monitor_path', type=str, required=True, help='save location for training data')
    parser.add_argument('--inter_param', type=float, required=True, help='interpolation parameter (alpha) used in the averaging process')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print('env_a:', args.env_a)
    print('env_b:', args.env_b)
    print('seed_init:', args.seed_init)
    print('seed_env:', args.seed_env)
    #print('model_path:', args.model_path)
    #print('monitor_path:', args.monitor_path)
    print('inter_param:', args.inter_param)
    
    set_random_seed(args.seed_init, using_cuda=True)
    
    TOTAL_STEPS=24_985_600
    MERGE_INTERVALL=16384
    # use same schema for logs and model save directory
    # monitor_path = './monitor/train/'+args.env+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    # model_path = './models/'+str(args.env)+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    #tb_path_a = './tensorboard/'+str(args.env_a)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
   # tb_path_b = './tensorboard/'+str(args.env_b)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    model_path_a = './models/test1a'
    model_path_b = './models/test1b'
    monitor_path_a = './monitortest1_a'
    monitor_path_b = './monitortest1_b'

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
                #tensorboard_log=tb_path_a,
                verbose=1)
    
    model_b = PPO('CnnPolicy',
                venv_b, 
                learning_rate=5e-4,
                n_steps=256,
                batch_size=2048,
                n_epochs=3,
                gamma=0.999,
                gae_lambda=0.95,
                ent_coef=0.01,
               # tensorboard_log=tb_path_b,
                verbose=1)
    
    for step in range(1, TOTAL_STEPS+1):
        model_a.learn(total_timesteps=1)
        model_b.learn(total_timesteps=1)

        if step % MERGE_INTERVALL == 0:
            # Merge model parameters every MERGE_INTERVALL steps (parameters are updated every 16384 steps)
            updated_params = gitrebasin(model_a.policy.state_dict(), model_b.policy.state_dict(), args.inter_param)
            model_a.policy.load_state_dict(updated_params)
            model_b.policy.load_state_dict(updated_params)

    model_a.save(model_path_a)
    model_b.save(model_path_b)

if __name__ == '__main__':
    main()