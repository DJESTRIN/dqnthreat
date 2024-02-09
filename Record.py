#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record
Used to save recordings of weights, biases and neural activity as agents progress into environment

Final Data Frame Layout:

AgentID Episode Step LayerID NeuronID WeightID ThreatBoolean NeuronalActivityValue NeuronalBiasValue WeightValue 
1234,1,1,fc1,1,1,0,7.02,1,0.001     
1234,1,1,fc1,1,2,0,7.02,1,0.002

"""
import ipdb
import numpy as np
import os,glob
import csv
from PIL import Image
import pandas as pd
from collections import defaultdict
import subprocess

class Record():
    def __init__(self,seed,output_dir,run_ilastik_file,ilastik_project):
        self.seed=seed
        self.run_ilastik_file=run_ilastik_file
        self.project_file=ilastik_project
        
        output_dir = output_dir+'/model_data/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Delete previous data in output_dir
        previous_files = glob.glob(output_dir+'**/*.npy')
        for file in previous_files:
            os.remove(file)
            
        self.output_dir=output_dir
        self.imageCounter = 0
        
    def grab_w_n_b(self,agent,episode):
        """ Saves weight and bias information for each agent as a npy file """
        for name, param in agent.policy_network.named_parameters():
            #ipdb.set_trace()
            if 'fc' in name:
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'
                if not os.path.exists(output_name):
                    os.mkdir(output_name)
                output_name=self.output_dir+'/'+name.replace('.','_')+'/'+str(self.seed)+'_'+str(episode)+"_"+name.replace('.','_')+'.npy'
                np.save(output_name,param.cpu().detach().numpy())
    
    def custom_sort(self,filename):
        try:
            _,_,_,_,number,_,_,_=filename.split('_')
        except:
            ipdb.set_trace()
        return int(number)
    
    def concat_w_n_b(self):
        """ Concatenates each episode's weights and biases for current agent """
        subfolders=glob.glob(self.output_dir+'/*')
        for folder in subfolders:
            for i,file in enumerate(sorted(glob.glob(folder+'/*.npy'),key=self.custom_sort)):
                if i==0:
                    fmat=np.load(file)
                    fmat=fmat[...,np.newaxis]
                    os.remove(file)
                else:
                    mat_oh=np.load(file)
                    mat_oh=mat_oh[...,np.newaxis]
                    os.remove(file)
                    fmat=np.concatenate((fmat, mat_oh),axis=-1) 
            output_name=folder+'/concat.npy'
            np.save(output_name,fmat)

    def activation_hook(self, agent, input, output):
        """ Save neural activity 
        Parameters
        ----------
        inst : torch.nn.Module
            The layer we want to attach the hook to.
        inp : tuple of torch.Tensor
            The input to the `forward` method.
        out : torch.Tensor
            The output of the `forward` method.
        """
        output_name=self.output_dir+'/'+'activations'+'.csv'
        with open(output_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(output.tolist())
        print("This is working!")
        

    def add_activation_hook(self, agent):
        # module is essentially the layer
        layerList = []
        for name, module in agent.named_modules():
            module.register_forward_hook(self.activation_hook)
            layerList.append(name)
            print(f"The name is:{name} and the module is {module}")

        output_name=self.output_dir+'/'+'activations'+'.csv'
        
        # Adding first row with layer names
        # I chose w to reset the file
        with open(output_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(layerList)
            
    def record_observation(self, observation, episode):
        path = f'{self.output_dir}/Episode{episode}'
        if not os.path.exists(path):
            os.mkdir(path)
        self.current_imagefilename=f'{path}/Image{self.imageCounter}.png'
        Image.fromarray(observation).save(self.current_imagefilename)
        self.segment_observation(self)
        
    def segment_observation(self):
        """ Run ilastik on current image via subprocess command """
        project_file='--project='+self.project_file
        image_file='--raw_data='+self.current_imagefilename
        subprocess.run([self.run_ilastik_file,'--headless','--output_format=numpy',project_file,image_file])
        self.classify_observation(self)
        
    def classify_observation(self):
        """ Determine whether threat is in current frame """  
        if self.game_name=="ALE/DemonAttack-v5":
           #inport ilastik segmented numpy file
           segmentation=np.load(self.run_ilastik_file)
           
           # If 
           ipdb.set_trace()
           ThreatBoolean=0

        else:
            raise TypeError("Game does not match ilastik file")
            
        return ThreatBoolean
    
    def buildDF(self):
        # Organizes each folder by its layer name
        # This assumes that each folder is named by "layer_bias" or "layer_weight"
        # We create a dictionary where key is the layer
        # Each value is an array with two folder names, the first being the bias and second being weights
        folderDict = defaultdict(list)
        subfolders=glob.glob(self.output_dir+'/*')

        for folder in subfolders:
            split_string = folder.split('_')
            if split_string[-1] in ['bias', 'weight']:
                folderDict[tuple(folder.split('_')[:-1])].append(folder)
        
        df = pd.DataFrame()

        for key, folders in folderDict.items():
            biasFolder, weightFolder = folders
            # Makes sure that the variables are correct
            if 'weight' not in weightFolder:
                weightFolder, biasFolder = biasFolder, weightFolder

            data = np.load(weightFolder + '/concat.npy')
            bias_data = np.load(biasFolder + '/concat.npy')

            datashape = data.shape

            next_layer_idx, neuron_idx, episode_idx = np.indices(datashape)

            # reshaped_data = data.reshape(-1, datashape[2])
            reshaped_bias_data = bias_data.reshape(-1)

            dictionary = {
                "Episode": episode_idx.flatten() + 1,
                "Neuron id": neuron_idx.flatten() + 1,
                "Next Layer Neuron": next_layer_idx.flatten() + 1,
                "Layer": os.path.basename(''.join(key)),
                # "Weight Value": reshaped_data.flatten(),
                "Weight Value": data.flatten(),
                "Bias Value": np.repeat(reshaped_bias_data, datashape[1])
            }

            df2 = pd.DataFrame(dictionary)
            df = pd.concat([df, df2])
        return df
    
# To do --> 
    # Need weights at end of every episode for every neuron along with biases
    # Need recording of activity for each neuron to determien whether neuron is activated by threat
    #   Need Image at same time to determine when "threat" is present.