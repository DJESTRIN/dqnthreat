import random
import numpy as np
import glob
import matplotlib.pyplot as plt 
from matplotlib.image import imread as ir
from gym import ObservationWrapper
import gymnasium as gym
import ipdb

class AddNoiseToGym(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obsreal):
        try:
            obs,other=obsreal
            d1,d2,d3=obs.shape #Get observation dimensions
            noise=np.random.uniform(-1,1,(d1,d2,d3)) #Get an array of noise values [0,1) same shape as observation
        except:
            d1,d2=obs.shape #Get observation dimensions
            noise=np.random.uniform(-1,1,(d1,d2)) #Get an array of noise values [0,1) same shape as observation
        obs=obs+(15*noise)
        obs[obs>255]=255
        obs[obs<0]=0
        obsreal=(obs,other)
        return obs


class ChangeDifficulty():
    def __init__(self,difficulty):
        #Initial difficulty scalar, 0 means no change. S
        self.difficulty=float(difficulty)
        
    def update_difficulty(self,difficulty):
        # update difficulty if you would like during the game
        self.difficulty=float(difficulty)

    def modify_observation(self,observation):
        try:
            d1,d2,d3,d4=observation.shape #Get observation dimensions
            noise=np.random.uniform(-1,1,(d1,d2,d3,d4)) #Get an array of noise values [0,1) same shape as observation
        except:
            d1,d2=observation.shape #Get observation dimensions
            noise=np.random.uniform(-1,1,(d1,d2)) #Get an array of noise values [0,1) same shape as observation
        observation=observation+(self.difficulty*noise) #Update observation using difficulty scalar
        observation[observation>255]=255
        observation[observation<0]=0
        return observation
    

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
