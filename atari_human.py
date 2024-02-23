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
if __name__=='__main__':
    play(gym.make("CartPole-v0", render_mode="rgb_array"), keys_to_action={  
                                                "w": np.array([0, 0.7, 0]),
                                                "a": np.array([-1, 0, 0]),
                                                "s": np.array([0, 0, 1]),
                                                "d": np.array([1, 0, 0]),
                                                "wa": np.array([-1, 0.7, 0]),
                                                "dw": np.array([1, 0.7, 0]),
                                                "ds": np.array([1, 0, 1]),
                                                "as": np.array([-1, 0, 1]),
                                                }, noop=np.array([0,0,0]))

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

