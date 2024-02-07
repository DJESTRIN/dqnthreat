#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:10:45 2024

@author: dje4001
"""
import argparse
import random
import time
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,FireResetEnv,MaxAndSkipEnv,NoopResetEnv)
from random import randrange
from PIL import Image

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropdirectory", type=str, required=True,
        help="root directory for saving data/results")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args

def make_env(env_id, seed, idx, capture_video, run_name, difficulty):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        env.env.game_difficulty=difficulty
        env.seed()
        env.reset()

        return env
    return thunk


if __name__ == "__main__":
    args = parse_args()
    results_directory=args.dropdirectory
    if results_directory[-1:]!='/':
        results_directory+='/' # Make sure last value is a forward slash

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 354 + i, i, False, 'get_observations',0) for i in range(1)])
    start_time = time.time()

    obs, _ = envs.reset(seed=354)
    for global_step in range(6000):
        actions = np.array([envs.single_action_space.sample() for _ in range(1)])
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        current_imagefilename=f'{args.dropdirectory}/Image{global_step}.png'
        Image.fromarray(next_obs).save(current_imagefilename)
        
    envs.close()
    

