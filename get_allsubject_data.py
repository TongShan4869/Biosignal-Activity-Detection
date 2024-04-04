# %%
import numpy as np
import pandas as pd
import BiosignalsMetadata
import os
import mne
# %% Load file
dataset_path = data_path = "../dataset/"
subject_list = BiosignalsMetadata.participants
sensor = 'emg:Left Bicep'
device = BiosignalsMetadata.col_to_device[sensor]
device = device.replace(' ','_').lower()
# %%
df = pd.DataFrame(columns=['subject','sensor','data','label'])
for subject in subject_list:
    # load data
    data_path = dataset_path + subject  + '/'+ device + '.edf'
    # check if file exist
    if not os.path.isfile(data_path):
        continue
    else:
        raw = mne.io.read_raw_edf(data_path, preload=True)
        # pick only the channel
        raw = raw.pick(['emg']) # pick channel according to the sensor of interest
        _, times = raw[:]
        # get data from annotation
        for i in range(len(raw.annotations)):
            label = raw.annotations[i]['description']
            if raw.annotations[i]['onset'] + raw.annotations[i]['duration'] > times[-1]: # if annotation duration exceed recording length
                # Remake a annotation and trim the duration
                fix_annotation = mne.Annotations(onset=raw.annotations[i]['onset'],
                                                duration=times[-1] - raw.annotations[i]['onset'], # trim the duration to the end of the recording
                                                description=raw.annotations[i]['description'],
                                                orig_time=raw.annotations[i]['orig_time'])
                # fetch data into array
                data = raw.copy().crop_by_annotations(fix_annotation)[0].get_data()
            else:
                data = raw.copy().crop_by_annotations(mne.Annotations(**raw.annotations[i]))[0].get_data()
            # save data
            df = df._append({'subject':subject,
                            'sensor':'emg',
                            'data':data,
                            'label':label}, ignore_index=True)

df.to_pickle(dataset_path+'emg_allsubject_data.pkl')
print("file saved...")
# %%
df.head(10)
# %%
