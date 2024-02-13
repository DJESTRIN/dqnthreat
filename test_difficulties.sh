#!/bin/bash
output_directory=$1
mkdir -p $output_directory

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    python3 dqn_atari_stripped.py \
        --seed $RANDOMSEED \
        --dropdirectory $output_directory \
        --exp-name easy$RANDOMSEED \
        --difficulty 0 \
        --track \
        --capture-video \
        --env-id ALE/SpaceInvaders-v5 \
        --agenttype vanilla \
        --buffer-size 400000 \
        --save-model

done

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    python3 dqn_atari_stripped.py \
        --seed $RANDOMSEED \
        --dropdirectory $output_directory \
        --exp-name easy$RANDOMSEED \
        --difficulty 1 \
        --track \
        --capture-video \
        --env-id ALE/SpaceInvaders-v5 \
        --agenttype vanilla \
        --buffer-size 400000 \
        --save-model

done
