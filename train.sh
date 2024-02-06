seed=$1
source ~/.bashrc
conda activate ataridqn

# Train Agent on Easy
#python3 dqn_atari_stripped.py --seed $seed --exp-name choppereasy --difficulty 0 --track --capture-video --env-id ALE/ChopperCommand-v5 --total-timesteps 3333333 --buffer-size 400000 --save-model

# Test Agent on Hard (no Learning)
python3 dqn_atari_stripped.py --seed $seed --exp-name chopperhard --difficulty 1 --track --capture-video --env-id ALE/ChopperCommand-v5 --total-timesteps 3333333 --buffer-size 400000 --save-model

# Test Agent on Hard (With learning) --- Need to import data from experience replay
#python3 dqn_atari.py --exp-name $agent_name --track --capture-video --env-id ALE/MsPacman-v5 --total-timesteps 5000000 --buffer-size 400000 --save-model



