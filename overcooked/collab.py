from misc.game.testplay import TestPlay
from misc.game.game import Game
import numpy as np
import argparse
import gym
import overcooked

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=35, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=True, help="Return observations as images (instead of objects)")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    return parser.parse_args()

def make_env(env_id, arglist, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, arglist=arglist)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    arglist = parse_arguments()
    env_id = "overcooked:overcookedEnv-v0"
    num_cpu = 4  # Number of processes to use
    # env = SubprocVecEnv([make_env(env_id, arglist, i) for i in range(num_cpu)])

    env = gym.envs.make("overcooked:overcookedEnv-v0", arglist=arglist)
    env.reset()
    # check_env(env)
    # # game = TestPlay(env.world, env.sim_agents, env)
    # # game.on_execute()
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[200, 100, 50, 10, 5])
    # # env = gym.make(env_id, total=10, good=3)
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=300000)

    # actions = [3,3,3,1,2,2,2,1,3,3,3,3,2,2,2,2,2,2,2]
    # for idx in range(len(actions)):
    #     action = actions[idx]
    #     new_obs, reward, done, info = env.step(action)
    #     print(new_obs)
    #     # game.on_execute()
    #     if done:
    #         break
    env.close()
