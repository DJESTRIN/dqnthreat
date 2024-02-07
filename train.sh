#!/bin/bash

#collect inputs into bash file
seed=$1
output_directory=$2
atari_game=$3
agent_type=$4
agent_name=$5

source ~/.bashrc
conda activate ataridqn

#Run the difficulty test

# Train Agent on Easy
python3 dqn_atari_stripped.py \
	--seed $seed \
	--dropdirectory $output_directory \
	--exp-name $agent_name \
	--difficulty 0 \
	--track \
	--capture-video \
	--env-id $atari_game \
	--agenttype $agent_type \
	--buffer-size 400000 \
	--save-model

# Test Agent on Hard (learning starts eventually)
python3 dqn_atari_stripped.py \
        --seed $seed \
        --dropdirectory $output_directory \
        --exp-name $agent_name \
        --difficulty 1 \
        --track \
        --capture-video \
        --env-id $atari_game \
        --agenttype $agent_type \
        --buffer-size 400000 \
        --save-model
