#!/bin/bash
OUTPUTDIR=$1 #real
OUTPUTDIR2=$2 #junk
OUTPUTDIR3=$3 #random
# Need to add this to train_atari.py
# Run sbatch
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test --mem=100G --partition=scu-gpu --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn.sh NoJunk not_random_agent dif_test_off $RANDOMSEED $OUTPUTDIR"
done

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test --mem=100G --partition=scu-gpu --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn.sh junk not_random_agent dif_test_off $RANDOMSEED $OUTPUTDIR2"
done

for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test --mem=100G --partition=scu-gpu --gres=gpu:1 --mail-type=BEGIN,END,FAIL --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu --wrap="bash run_dqn.sh NoJunk random_agent dif_test_off $RANDOMSEED $OUTPUTDIR3"
done

sbatch --mem=5G --partition=scu-cpu --dependency=singleton --job-name=DQN_test --wrap="bash change_ownership.sh"

exit




# Settings for run_dqn.sh script
# junk    -- will create junk agents
# random_agent -- will create random agents
# dif_test_on -- will create a difficulty test simulating stress
