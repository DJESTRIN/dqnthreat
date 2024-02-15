#!/bin/bash
output_directory=$1
mkdir -p $output_directory

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=stitch_files --mem=50G --partition=scu-cpu --mail-type=BEGIN,END,FAIL \
        --mail-user=dje4001@med.cornell.edu \
        --wrap=" source ~/.bashrc; conda activate ataridqn; python3 dqn_atari_stripped.py \
        --seed $RANDOMSEED \
        --dropdirectory $output_directory \
        --exp-name easy$RANDOMSEED \
        --difficulty 0 \
        --capture-video \
        --env-id ALE/SpaceInvaders-v5 \
        --agenttype vanilla \
        --buffer-size 400000 \
        --save-model"

done

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=stitch_files --mem=50G --partition=scu-cpu --mail-type=BEGIN,END,FAIL \
        --mail-user=dje4001@med.cornell.edu \
        --wrap=" source ~/.bashrc; conda activate ataridqn; python3 dqn_atari_stripped.py \
        --seed $RANDOMSEED \
        --dropdirectory $output_directory \
        --exp-name hard$RANDOMSEED \
        --difficulty 1 \
        --capture-video \
        --env-id ALE/SpaceInvaders-v5 \
        --agenttype vanilla \
        --buffer-size 400000 \
        --save-model"

done
