#!/bin/bash
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
