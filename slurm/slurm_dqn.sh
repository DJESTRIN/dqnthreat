#!/bin/bash

OUTPUTDIR=$1 #real
OUTPUTDIR2=$2 #junk
OUTPUTDIR3=$3 #random

# Run sbatch
# Vanilla Agent
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test \
	--mem=30G \
	--partition=scu-cpu \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu \
	--wrap="bash train.sh $RANDOMSEED $OUTPUTDIR ALE/SpaceInvaders-v5 vanilla $RANDOMSEED"
done

# Junk Agent
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test \
        --mem=30G \
        --partition=scu-cpu \
        --mail-type=BEGIN,END,FAIL \
        --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu \
        --wrap="bash train.sh $RANDOMSEED $OUTPUTDIR2 ALE/SpaceInvaders-v5 junk $RANDOMSEED"
done

# Random Agent
for i in {1..10}
do
    CURRENTEPOCTIME=`date +%s`
    RANDOMSEED=$(($CURRENTEPOCTIME + $i))
    sbatch --job-name=DQN_test \
        --mem=30G \
        --partition=scu-cpu \
        --mail-type=BEGIN,END,FAIL \
        --mail-user=dje4001@med.cornell.edu,anp4047@med.cornell.edu \
        --wrap="bash train.sh $RANDOMSEED $OUTPUTDIR3 ALE/SpaceInvaders-v5 random $RANDOMSEED"
done

# Change ownership so both of us can see results
sbatch --mem=5G --partition=scu-cpu --dependency=singleton --job-name=DQN_test --wrap="bash change_ownership.sh"

exit

