from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
event_acc = EventAccumulator('/athena/listonlab/scratch/anp4047/difftest4/runs/easy1708461779__1708461779/events.out.tfevents.1708461792.c0009.2543542.0')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())
df=pd.DataFrame(event_acc.Scalars('charts/episodic_return'))
print(df)