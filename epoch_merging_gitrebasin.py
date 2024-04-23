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
    parser.add_argument('--seed_init_a', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_init_b', type=int, required=True, help='seed for model weight initialization')
    parser.add_argument('--seed_env', type=int, required=True, help='seed for environment level generation')
    #parser.add_argument('--model_path', type=str, required=True, help='save location for model after training')
    #parser.add_argument('--monitor_path', type=str, required=True, help='save location for training data')
    parser.add_argument('--inter_param', type=float, required=True, help='interpolation parameter (alpha) used in the averaging process')
    parser.add_argument('--merge_intervall', type=float, required=True, help='sets after how many parameter updates the models are merged')
    return parser.parse_args()


def main():
    args = parse_args() 
    print('env_a:', args.env_a)
    print('env_b:', args.env_b)
    print('seed_init_a:', args.seed_init_a)
    print('seed_init_b:', args.seed_init_b)
    print('seed_env:', args.seed_env)
    #print('model_path:', args.model_path)
    #print('monitor_path:', args.monitor_path)
    print('inter_param:', args.inter_param)
    print('merge_intervall:', args.merge_intervall)
    
    set_random_seed(args.seed_init_a, using_cuda=True)
    
    TOTAL_UPDATES = 1530
    MERGE_INTERVALL = args.merge_intervall
    # use same schema for logs and model save directory
    # monitor_path = './monitor/train/'+args.env+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    # model_path = './models/'+str(args.env)+'_'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
    #tb_path_a = './tensorboard/'+str(args.env_a)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
   # tb_path_b = './tensorboard/'+str(args.env_b)+'/'+'init'+str(args.seed_init)+'_'+'env'+str(args.seed_env)
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
        model_a.learn(total_timesteps=16384, reset_num_timesteps=False)
        model_b.learn(total_timesteps=16384, reset_num_timesteps=False)
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

if __name__ == '__main__':
    main()