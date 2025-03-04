from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tqdm
import ipdb

search_string = '/athena/listonlab/scratch/anp4047/difficulty_tests/diff0vdiff100//runs/**/events*'
tbfiles = glob.glob(search_string)

plt.figure()
for file in tqdm.tqdm(tbfiles):
    event_acc = EventAccumulator(file)
    event_acc.Reload()
    try:
        df=pd.DataFrame(event_acc.Scalars("charts/episode_length"))
        df=df.rolling(1000).mean()
        if "easy" in file:
            #ipdb.set_trace()
            plt.plot(df["step"],df["value"],'g')
        elif "hard" in file:
            plt.plot(df["step"],df["value"],'r')
    except:
        print(f'{file} was skipped')

plt.savefig('/home/fs01/dje4001/dqnthreat/diff_test_episode_lengths.jpg')
