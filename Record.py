#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record Neural Network
(1) Collects NN activations,weights and biases. 
(2) Applies ilastik pixel classification to classify observation
"""
import ipdb
import numpy as np
import os, glob
import csv
from PIL import Image
import pandas as pd
from collections import defaultdict
import subprocess
import time
import math

# Super Logger is a class to manage ActivationLoggers
class SuperLogger():
    # Holds a list of loggers, and uses the outputs to create a list to place into a csv
    def __init__(self, csvFilePath):
        self.loggers = []
        self.enemy_boolean = None
        self.attack_boolean = None
        self.agent_boolean = None
        self.enemy_distance = None
        self.attack_distance = None
        self.csvFilePath = csvFilePath
    
    # addLogger returns the logger, allowing us to directly register the hook 
    def addLogger(self):
        logger = ActivationLogger(self)
        self.loggers.append(logger)
        return logger
    
    def setThreatInfo(self, enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance):
        self.enemy_boolean = enemy_boolean
        self.attack_boolean = attack_boolean
        self.agent_boolean = agent_boolean
        self.enemy_distance = enemy_distance
        self.attack_distance = attack_distance
        

    def checkAddToCSV(self):
        # If all loggers have an output and observation is recorded, we can place into the csv
        # After placing into csv, we can reset all outputs
        newRow = []
        for logger in self.loggers:
            if logger.output == None:
                return
            else:
                newRow.append(logger.output)

        
        if self.enemy_boolean == None:
            return
        if self.attack_boolean == None:
            return
        if self.agent_boolean == None:
            return
        if self.enemy_distance == None:
            return
        if self.attack_distance == None:
            return
       
        newRow.append(self.enemy_boolean)
        newRow.append(self.attack_boolean)
        newRow.append(self.agent_boolean)
        newRow.append(self.enemy_distance)
        newRow.append(self.attack_distance)

        with open(self.csvFilePath, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(newRow)
            self.setAllLoggers(None)
            self.enemy_boolean = None
            self.attack_boolean = None
            self.agent_boolean = None
            self.enemy_distance = None
            self.attack_distance = None

    def setAllLoggers(self, value):
        for logger in self.loggers:
            logger.output = value
        
# Activation Logger allows us to hold onto the output value
class ActivationLogger:
    def __init__(self, superLogger):
        self.superLogger = superLogger
        self.output = None

    def __call__(self, module, input, output):
            self.output = output.tolist()
            self.superLogger.checkAddToCSV()

# Analyze Observation, pulls relevant data from current observation for data frame             
class AnalyzeObservation():
    def __init__(self,ilastik_dir,ilp_file,output_dir):
        self.ilastik_dir=ilastik_dir
        self.ilp_file=ilp_file
        self.output_dir=output_dir

    def parsesegmentation(self,array):
        """ Calculate the presence of threat. Then calculate agent distance from threat
        (1) Background -- image data not relevant
        (2) Agent -- the origin of the agent that is in environment
        (3) Enemy -- the competitor which produces an attack 
        (4) Attack -- the mechanism by which agent dies if touched (eg. bullet, sword, etc)
        """
        if array.shape[2]<4:
            raise TypeError("ilastik numpy output should have 4 slices")

        #Binerize array based on threshold 0.9
        array[array<0.9]=0
        array[array>0.9]=1

        #Parse out important variables
        background=array[:,:,0]
        agent=array[:,:,1]
        enemy=array[:,:,2]
        attack=array[:,:,3]

        # Boolean values whether enemy or attack are present in observation
        enemy_boolean = np.sum(enemy)>0
        attack_boolean = np.sum(attack)>0
        agent_boolean = np.sum(agent)>0

        # Calculate distances
        # distance between enemy and agent 
        if agent_boolean:
            y_agent,x_agent=np.where(agent==1)
            y_agent,x_agent=np.mean(y_agent),np.mean(x_agent) #average coordinate of agent

        if enemy_boolean and agent_boolean:
            y_enemy,x_enemy=np.where(enemy==1)
            y_enemy,x_enemy=np.mean(y_enemy),np.mean(x_enemy) #average coordinate of enemy

            # Calculate distance between agent and enemy
            enemy_distance = self.distance(x_agent,y_agent,x_enemy,y_enemy)
        else:
            enemy_distance = np.nan

        # distance between attack and agent
        if attack_boolean and agent_boolean:
            y_attack,x_attack=np.where(attack==1)
            y_attack,x_attack=np.mean(y_attack),np.mean(x_attack) #average coordinate of enemy

            # Calculate distance between agent and enemy
            attack_distance = self.distance(x_agent,y_agent,x_attack,y_attack)
        else:
            attack_distance = np.nan

        return enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance

    def distance(self,Px,Py,Qx,Qy):
        """ Get distance. Cite: https://www.geeksforgeeks.org/python-math-dist-method/ """
        return math.dist([Px,Py],[Qx,Qy])
    
    def __call__(self, observation):
        """ Get relevant data from observation
         cite: https://www.ilastik.org/documentation/basics/headless """

        # Clock time to determine code effeciency -- are calls to ilastik too slow?
        start_time = time.time()

        # Create temp folder. 
        tmppath=f'{self.output_dir}/ilastiktmp/'
        if not os.path.exists(tmppath):
            os.mkdir(tmppath)
        
        # Save current image
        timestr = time.strftime("%Y%m%d-%H%M%S")    # Provide unique key 
        image_file=f'{tmppath}/ImageOH{timestr}.png'
        Image.fromarray(observation).save(image_file)

        # Build ilastik command
        command = f'{self.ilastik_dir} --headless --project={self.ilp_file} --export_source="Probabilities" --output_format=numpy {image_file}'
        subprocess.call(command, shell=True)

        # Load numpy file
        files=glob.glob(f'{tmppath}/*.npy')

        if len(files)>2: # Only a single numpy file should be in the temporary directory at a time
            raise TypeError("Multiple numpy files found, analysis will be wrong")

        npfile=files[0]
        array = np.load(npfile)

        # Parse the numpy array
        enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance = self.parsesegmentation(array)

        # garbage collection: delete image and corresponding numpy file
        os.remove(image_file)
        os.remove(npfile)

        #calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.3f} seconds")
        return enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance   

class Record():
    def __init__(self,seed,output_dir,ilastik_dir,ilp_file):
        self.seed=seed
        
        output_dir = output_dir+'/model_data/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        # Delete previous data in output_dir
        previous_files = glob.glob(output_dir+'**/*.npy')
        for file in previous_files:
            os.remove(file)
            
        self.output_dir=output_dir

        self.imageCounter = 0

        self.activationHookFiles = []

        self.superLogger = None

        # created threat detector object.  
        self.threat_detector=AnalyzeObservation(ilastik_dir,ilp_file,output_dir)
        
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

    def classify_observation(self,observation):
        # Get relevant threat information from observation
        enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance = self.threat_detector(observation)

        # We need to account for the case that superLogger has not been initialized yet
        if self.superLogger != None:
            self.superLogger.setThreatInfo(enemy_boolean, attack_boolean, agent_boolean, enemy_distance, attack_distance)

    def add_activation_hook(self, agent):
        # module is essentially the layer
        # Keeps track of layer name and description to add to top of csv
        layerList = []
        outputPath=self.output_dir+'/'+'activations'+'.csv'
        self.superLogger = SuperLogger(outputPath)
        
        # superLogger will add loggers in correct order
        for name, module in agent.named_modules():
            name = name if name != "" else "no_name_network"
            module.register_forward_hook(self.superLogger.addLogger())
            layerList.append(name)
        
        layerList.append('enemy_boolean')
        layerList.append('attack_boolean')
        layerList.append('agent_boolean')
        layerList.append('enemy_distance')
        layerList.append('attack_distance')

        # Adding first row with layer names
        # I chose w to reset the file
        with open(outputPath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(layerList)
            
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
    
    def activationHookDF(filepath):
        df = pd.read_csv(filepath)
        # By default, all of the "numpy" values in the dataframe have an additional dimension
        # e.g. an array of 9 elements is (1,9)
        # We can squeeze it
        
        # The first column is generally the index, so we can remove it
        orig_columns = df.columns[1:]

        # Converts string array into numpy, and removes dimension
        df[orig_columns] = df[orig_columns].applymap(lambda x: np.squeeze(np.array(eval(x)), axis=0))

        for col in orig_columns:
            # We can use the shape of the first element as reference for all elements in the Series
            for i in range(df[col].iloc[0].shape[0]):
                df[f"{col}_{i}"] = df[col].apply(lambda x: x[i])

            df.drop(columns=[col], inplace=True)
            
        return df