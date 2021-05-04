from misc.game.testplay import TestPlay
from misc.game.game import Game
import numpy as np
import argparse
import gym
import overcooked


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=True, help="Return observations as images (instead of objects)")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_arguments()
    env = gym.envs.make("overcooked:overcookedEnv-v0", arglist=arglist)
    env.reset()
    # game = TestPlay(env.world, env.sim_agents, env)
    # game.on_execute()
    actions = [3,3,3,1,2,2,2,1,3,3,3,3,2,2,2,2,2,2,2]
    for idx in range(len(actions)):
        action = actions[idx]
        new_obs, reward, done, info = env.step(action)
        # game.on_execute()
        if done:
            break
    env.close()
