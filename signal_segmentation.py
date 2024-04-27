# %%
import numpy as np
import pandas as pd
import BiosignalsMetadata as bsm
from matplotlib import pyplot as plt

# %% parameters
fs = 500
seg_len = 10 # 10 second length
subject_list = bsm.participants
dataset_path = data_path = "../csv/"
segment_temp = pd.read_csv("segments_template.csv")
segment_temp_ref = pd.DataFrame(columns=list(segment_temp.columns[2:6])+bsm.columns)
del segment_temp
# %%
for subject in subject_list:
    df = pd.read_csv(dataset_path+subject+'.csv')
    # get real activities excluding (blank)/(transition)
    act_list = [act for act in bsm.activities[subject] if not act.startswith('(')]
    for a in act_list:
        act = bsm.act_to_category[a]
        time_range = bsm.activity_ranges[subject][a][0]
        duration = (time_range[1]-time_range[0])/fs
        # This in only for "03FH" Baseline -> change to full 10 s
        if duration < 10:
            time_range = (time_range[0], time_range[0] + int(10*fs))
            duration = (time_range[1]-time_range[0])/fs
        n_seg = int(duration//10)
        for i in range(n_seg):
            modal_list = []
            for c in segment_temp_ref.columns[4:]:
                if c not in df.columns:
                    modal_list += [False]
                elif c in df.columns:
                    data = df[c][time_range[0]+i*10*fs:time_range[0]+(i+1)*10*fs]
                    if pd.isna(data).any():
                        modal_list += [False]
                    else:
                        modal_list += [True]
            segment_temp_ref.loc[len(segment_temp_ref)] = [subject, time_range[0]+i*10*fs, time_range[0]+(i+1)*10*fs,act]+modal_list

segment_temp_ref.to_csv("segments_template_refine.csv")
# 
# %%
segment_temp_ref_copy = segment_temp_ref.copy()
segment_temp_ref_copy.replace(False, np.nan, inplace=True)
segment_temp_ref_copy.dropna(inplace=True)
segment_temp_ref_copy['Activity'].value_counts()
# %%
segment_temp_ref_copy = segment_temp_ref[segment_temp_ref[segment_temp_ref.columns[4:]].any(axis=1)]
segment_temp_ref_copy.to_csv("segments_template_refine_at_least_one.csv")
# %%
