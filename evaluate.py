from procgen import ProcgenEnv
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack, # if wanting to use framestack sometime, currently not used
)
import gym
from stable_baselines3.common.utils import set_random_seed


def spectate(env, num_episodes, seed, model):
    venv = gym.make('procgen:procgen-'+env+'-v0', render_mode='human', distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed, use_backgrounds=False)
    
    model = PPO.load(model, env=venv)

    obs = venv.reset()
    num_episodes = num_episodes
    ep_counter = 0

    while ep_counter < num_episodes:
        action, _ = model.predict(obs)
        obs, _, done, _ = venv.step(action)
        if done:
            obs = venv.reset()
            ep_counter += 1
            print('completed episodes:', ep_counter)

    venv.close()


def test_random_agent(env, num_episodes, seed, logdir):
    venv = ProcgenEnv(num_envs=64, env_name=env, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed, use_backgrounds=False)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=logdir)
    _ = venv.reset()
    ep_counter = 0
    while ep_counter < num_episodes:
        action = np.array([venv.action_space.sample() for _ in range(venv.num_envs)])
        _, _, dones, _ = venv.step(action)
        for done in dones:
            if done:
                ep_counter += 1
                if (ep_counter%1000)==0:
                    print('completed episodes:', ep_counter)
                if ep_counter >= num_episodes:
                    return 0
                

def test(env, num_episodes, seed, model, logdir):
    venv = ProcgenEnv(num_envs=64, env_name=env, distribution_mode='easy', num_levels=0, use_sequential_levels=False, 
                        restrict_themes=False, rand_seed=seed, use_backgrounds=False)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=logdir)
    
    model = PPO.load(model, env=venv, device="cuda")
    obs = venv.reset()
    ep_counter = 0
    while ep_counter < num_episodes:
        action, _ = model.predict(obs)
        obs, _, dones, _ = venv.step(action)
        for done in dones:
            if done:
                ep_counter += 1
                if (ep_counter%1000)==0:
                    print('completed episodes:', ep_counter)
                if ep_counter >= num_episodes:
                    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, required=True, help='specifies number of episodes to evaluate model (5000 used in thesis)')
    parser.add_argument('--seed', type=int, required=True, help='used for libraries and level generation')
    parser.add_argument('--mode', default='spectate', type=str, required=True, choices=['spectate', 'test', 'random'], help='spectate: watch model play; test: test model and save data; random: create and save data for random agent')
    parser.add_argument('--model', type=str, help='path to model to be evaluated (not required when evaluating random agent)')
    parser.add_argument('--logdir', type=str, help='save location for test data when using modes test or random')
    return parser.parse_args()


def main():
    args = parse_args()
    print('env:', args.env)
    print('seed:', args.seed)
    print('mode:', args.mode)
    print('num_episodes:', args.num_episodes)
    if args.model is not None:
        print('loading model:', args.model)

    set_random_seed(args.seed, using_cuda=True)
    
    if args.mode == 'spectate':
        spectate(args.env, args.num_episodes, args.seed, args.model)
    elif args.mode == 'test':
        print('saving monitor file in:', args.logdir)
        test(args.env, args.num_episodes, args.seed, args.model, args.logdir)
    else:
        print('saving monitor file in:', args.logdir)
        test_random_agent(args.env, args.num_episodes, args.seed, args.logdir)


if __name__ == '__main__':
    main()