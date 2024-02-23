# import argparse
# import os
# import random
# import time
# from distutils.util import strtobool
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from stable_baselines3.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,FireResetEnv,MaxAndSkipEnv,NoopResetEnv)
# from stable_baselines3.common.buffers import ReplayBuffer
# from torch.utils.tensorboard import SummaryWriter
# from random import randrange
# #import Record
# from ChangeDifficulty import ChangeDifficulty
# from gymnasium.utils.play import PlayableGame

import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
from ChangeDifficulty import AddNoiseToGym
import argparse
from gymnasium.wrappers import TransformObservation

parser = argparse.ArgumentParser()
parser.add_argument("--diff", type=int,required=True)
parser.add_argument("--mode", type=int,required=True)

if __name__=='__main__':
    args = parser.parse_args()
    env = gym.make("ALE/DemonAttack-v5", render_mode="rgb_array",difficulty=args.diff,mode=args.mode)
    env = TransformObservation(env, lambda obs: print(obs) )
    env.reset()
    # env = AddNoiseToGym(env)
    play(env,keys_to_action=None,zoom=3)



# def make_env(env_id, seed, idx, capture_video, run_name):
#     if capture_video and idx == 0:
#         env = gym.make(env_id, render_mode="human", difficulty=0)
#         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#     else:
#         env = gym.make(env_id, render_mode="human", difficulty=0)
   
#     #env = PlayableGame(env)
#     env.action_space.seed(seed)

#     env.seed()
#     env.reset()

#     return env
    
    

# if __name__=='__main__':
#     env = make_env('ALE/SpaceInvaders-v5', 57, 1, False, 'test')

