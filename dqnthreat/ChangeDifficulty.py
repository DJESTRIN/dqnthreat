import random
import numpy as np
import glob, os
import matplotlib.pyplot as plt 
from matplotlib.image import imread as ir
from gym import ObservationWrapper
import gymnasium as gym
from gymnasium.spaces import Box
import ipdb
import cv2

class AddNoiseToObservation():
    def __init__(self,difficulty):
        #Initial difficulty scalar, 0 means no change. S
        self.difficulty=float(difficulty)
        self.episode=0
        
    def update_difficulty(self,difficulty):
        # update difficulty if you would like during the game
        self.difficulty=float(difficulty)

    def modify_observation(self,observation):
        # Adds noise to observation, thereby increasing difficulty.
        # If difficulty < 0, observation will be a blank screen as a control

        if self.difficulty>0:
            noise=np.random.uniform(-1,1,observation.shape) #Get an array of noise values [0,1) same shape as observation
            observation=observation+(self.difficulty*noise) #Update observation using difficulty scalar
            observation[observation>255]=255
            observation[observation<0]=0
        else:
            #Zero observation to black screen as a control
            observation = observation - observation
        return observation
    
    def capture_video(self,observation,terminated,drop_folder):
        if not os.path.exists(drop_folder):
            os.makedirs(drop_folder)

        if self.episode==0:
            self.episode+=1
            self.filename=f'{drop_folder}episode{self.episode}.mp4'
            fps=10
            self.video=cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (observation.shape[2], observation.shape[3]), 0)

        if terminated:
            #Generate new video per episode
            self.video.release() # Release the previous video
            self.episode+=1
            self.filename=f'{drop_folder}episode{self.episode}.mp4'
            fps=10
            self.video=cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (observation.shape[2], observation.shape[3]), 0)
 
        # Modify observation
        obs = np.squeeze(observation)
        obs = obs[0,:,:]
        obs = obs.astype(np.uint8) #Cannot pass float into cv2 write, must be defined pixels

        # Write observation into video
        self.video.write(obs)
   

def plot_example_difficulties(image_path):
    search_string=image_path+'*.png'
    example_images=glob.glob(search_string)
    for image in example_images:
        im=ir(image) #image read
        set_diff=ChangeDifficulty(0) #Set up object
        f, axs = plt.subplots(2,5)
        f.set_figheight(15)
        f.set_figwidth(15)

        for j in range(10):
            set_diff.update_difficulty(j/10)

            if j<5:
                row=0
                col=j
            else:
                row=1
                col=j-5

            axs[row,col].imshow(set_diff.modify_observation(im),cmap="gray")
            axs[row, col].set_title(f'Difficulty = {j/10}')
            axs[row, col].axis("off")
        savestring=image_path+'noisyfig.JPG' # only saving one image for now
        f.savefig(savestring)

if __name__=='__main__':
    plot_example_difficulties('D:\\dqnthreat\\')
