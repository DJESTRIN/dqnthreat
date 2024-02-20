
import random
import numpy as np

class ChangeDifficulty():
    def __init__(self,difficulty):
        #Initial difficulty scalar, 0 means no change. S
        self.difficulty=difficulty
        
    def update_difficulty(self,difficulty):
        # update difficulty if you would like during the game
        self.difficulty=difficulty

    def modify_observation(self,observation):
        d1,d2,d3,d4=observation.shape() #Get observation dimensions
        noise=np.random.rand(d1,d2,d3,d4) #Get an array of noise values [0,1) same shape as observation
        observation=observation+(self.difficulty*noise) #Update observation using difficulty scalar
        observation[observation>255]=255
        observation[observation<0]=0
        return observation
