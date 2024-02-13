#!/bin/bash
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
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

done
