
import random
import ipdb

class ChangeDifficulty():
    def __init__(self,difficulty):
        self.difficulty=difficulty
        
    def update_difficulty(self,difficulty):
        self.difficulty=difficulty

    def modify_observation(self,observation):
        if self.difficulty==0:
            return observation
        
        else:
            ipdb.set_trace()