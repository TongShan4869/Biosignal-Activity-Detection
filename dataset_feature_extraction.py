# %% Import libs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from scipy.signal import hilbert, butter, filtfilt, lfilter
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

import BiosignalsMetadata as bsm
import biosppy.signals as bsp
import neurokit2 as nk
import biobss

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# %% Data path
dataset_path = data_path = "../csv/"
fs = 500
# %% EMG feature extraction
def extract_low_env(sig, cutoff, fs, order):
    """Extract low envelope of the signal

    Prameters
    ----------
    sig : array
        input EMG signal
    cutoff : float
        cutoff frequency
    fs : float
        sampling frequency
    order : int
        order of the filter
    
    Returns
    ----------
    env : array
        envelope
    """
    # extract envelope
    sig_hilb = np.abs(hilbert(sig))
    # low pass envelope
    b, a = butter(N=order, Wn=cutoff/(0.5*fs), btype='low')
    env = lfilter(b, a, sig_hilb)
    return env
def calculate_entropy(probabilities):
    probabilities = probabilities[probabilities > 0]  # Filter zero probabilities to avoid log(0)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def normalized_power(sig,fs):
    # Frequency Domain
    t=np.arange(0,len(np.array(sig))/fs, 1/fs)
    # Perform FFT and compute the power spectral density (PSD)
    fft_vals = fft(sig)
    fft_power = np.abs(fft_vals)**2
    frequencies = fftfreq(len(fft_power), t[1] - t[0])
    # Consider only the positive frequencies
    positive_freqs = frequencies > 0
    fft_power = fft_power[positive_freqs]
    frequencies = frequencies[positive_freqs]
    # Normalize the power spectral density to create a probability distribution
    fft_power_normalized = fft_power / np.sum(fft_power)
    return fft_power, fft_power_normalized, frequencies

def emg_features(data, fs):
    # zero-mean
    data = data - np.mean(data)
    # statistical features
    feature_std = data.std()
    feature_skew = skew(data)
    feature_kurtosis = kurtosis(data)
    # extract envelope
    env = extract_low_env(data, 1, fs, 4)
    # raw normalized power
    raw_power, raw_norm_power, freq = normalized_power(data,fs)
    # raw entropy
    raw_entropy = calculate_entropy(raw_norm_power)
    # raw mean median frequency
    if (np.cumsum(raw_norm_power) <= (np.sum(raw_norm_power) / 2)).any():
        raw_median_freq = freq[np.cumsum(raw_norm_power) <= (np.sum(raw_norm_power) / 2)][-1]
    else:
        raw_median_freq = freq[0]
    raw_mean_freq = np.sum(freq * raw_norm_power) / np.sum(raw_norm_power)
    # env norm power
    env_power, env_norm_power, freq = normalized_power(env,fs)
    # env entropy
    env_entropy = calculate_entropy(env_norm_power)
    # 
    if (np.cumsum(env_norm_power) <= (np.sum(env_norm_power) / 2)).any():
        env_median_freq = freq[np.cumsum(env_norm_power) <= (np.sum(env_norm_power) / 2)][-1]
    else:
        env_median_freq = freq[0]
    env_mean_freq = np.sum(freq * env_norm_power) / np.sum(env_norm_power)
    # Norm env power spectral band
    band_len = 2
    n_bands = 5
    power_norm_band = []
    for k in range(n_bands):
        power_norm_band += [raw_norm_power[k*band_len:(k+1)*band_len].sum()]
    
    _, _, onsets = bsp.emg.emg(signal = data, sampling_rate = fs, show = False)
    emg_onsets = len(onsets)
    emg_onsets_reg = 0
    if emg_onsets > 2:
        emg_onsets_reg = 1 - np.std(np.diff(onsets), ddof = 1)/np.mean(onsets)
    return {
        'emg_std': feature_std,
        'emg_skew': feature_skew,
        'emg_kurtosis': feature_kurtosis,
        'emg_raw_entropy':  raw_entropy,
        'emg_raw_median_freq': raw_median_freq,
        'emg_raw_mean_freq': raw_mean_freq,
        'emg_env_entropy': env_entropy,
        'emg_env_median_freq': env_median_freq,
        'emg_env_mean_freq': env_mean_freq,
        'emg_power_band_1': power_norm_band[0],
        'emg_power_band_2':power_norm_band[1],
        'emg_power_band_3':power_norm_band[2],
        'emg_power_band_4':power_norm_band[3],
        'emg_power_band_5': power_norm_band[4],
        'emg_onsets': emg_onsets,
        'emg_onsets_reg': emg_onsets_reg,
    }
# %% 
def ppg_features(data, fs, prefix):
    filtered_data = biobss.preprocess.filter_signal(data, sampling_rate=fs, 
                                                      signal_type='PPG', method='bandpass')
    info=biobss.ppgtools.ppg_detectpeaks(sig=filtered_data, sampling_rate=fs, method='peakdet', delta=0.01, correct_peaks=True)
    locs_peaks=info['Peak_locs']
    peaks=filtered_data[locs_peaks]
    locs_onsets=info['Trough_locs']
    onsets=filtered_data[locs_onsets]
    features_all = biobss.ppgtools.get_ppg_features(sig=filtered_data, sampling_rate=fs,
                                            input_types=['cycle','segment'], 
                                            feature_domain={'cycle':['Time'],'segment':['time','freq','stat']},
                                            peaks_locs=locs_peaks, peaks_amp=peaks,
                                            troughs_locs=locs_onsets, troughs_amp=onsets,
                                            prefix = prefix)
    return features_all

# %% load template
segments_temp = pd.read_csv("segments_template_nonan.csv")
# delete row 2087-2093 -- no PPG finger data
segments_temp = segments_temp.drop(index=np.arange(2087, 2094))
segments_temp = segments_temp.reset_index(drop=True)

segments_temp_features = segments_temp.copy()
segments_temp_features.drop(segments_temp_features.columns[5:], axis=1, inplace=True)

# %% Loop through each segments
subject = "Subject"
for i in range(len(segments_temp)):
#for i in range(2094, len(segments_temp)):
    print(str(i)+" out of "+str(len(segments_temp)))
    if segments_temp.iloc[i]["Participant"] != subject:
        subject = segments_temp.iloc[i]["Participant"]
        df = pd.read_csv(dataset_path+subject+'.csv')

    # extract EMG features
    emg_data = df['emg:Left Bicep'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    emg_data = np.array(emg_data)
    emg_feature = emg_features(emg_data, fs)

    # extract PPG wrist features
    ppg_wrist_data = df['ppg:Left Wrist'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    ppg_wrist_data = np.array(ppg_wrist_data)
    ppg_wrist_feature = ppg_features(ppg_wrist_data, fs, prefix='ppg_wrist')

    # # extract PPG finger feature
    ppg_finger_data = df['ppg:Left index finger'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    ppg_finger_data = np.array(ppg_finger_data)
    ppg_finger_feature = ppg_features(ppg_finger_data, fs, prefix='ppg_finger')

    # extract ECG feature
    ecg_data = df['ecg:dry'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    ecg_data = np.array(ecg_data)
    rpeaks = biobss.ecgtools.ecg_detectpeaks(ecg_data, fs)
    ecg_feature = biobss.ecgtools.from_Rpeaks(ecg_data, rpeaks, fs, average = True)

    # extract EDA feature
    eda_data = df['eda:dry'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    eda_data = np.array(eda_data)
    filtered_eda_data = biobss.edatools.eda_filter.filter_eda(eda_data,fs)
    eda_features_ = biobss.edatools.eda_features.from_signal(filtered_eda_data, sampling_rate=fs)
    eda_feature = {"eda_" + key: value for key, value in eda_features_.items()}

    # extract ACC chest feature
    accCHx_data = df['acc_chest:x'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accCHx_data = np.array(accCHx_data)
    accCHy_data = df['acc_chest:y'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accCHy_data = np.array(accCHy_data)
    accCHz_data = df['acc_chest:z'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accCHz_data = np.array(accCHz_data)
    acc_chest_feature = biobss.imutools.get_acc_features(signals=[accCHx_data, accCHy_data,accCHz_data], signal_names=['accCHx','accCHy', 'accCHz'], sampling_rate=fs)

    # extract ACC wrist feature
    accWx_data = df['acc_e4:x'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accWx_data = np.array(accWx_data)
    accWy_data = df['acc_e4:y'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accWy_data = np.array(accWy_data)
    accWz_data = df['acc_e4:z'][segments_temp.iloc[i]['IRange_start']:segments_temp.iloc[i]['IRange_end']]
    accWz_data = np.array(accWz_data)
    acc_wrist_feature = biobss.imutools.get_acc_features(signals=[accWx_data, accWy_data,accWz_data], signal_names=['accWx','accWy', 'accWz'], sampling_rate=fs)

    # Adding feature into Dataframe
    all_features = {**emg_feature, **ppg_wrist_feature, **ppg_finger_feature, **ecg_feature, **eda_feature, **acc_chest_feature, **acc_wrist_feature}
    # Delete PPG finger
    # all_features = {**emg_feature, **ppg_wrist_feature, **ecg_feature, **eda_feature, **acc_chest_feature, **acc_wrist_feature}

    for ky, val in all_features.items():
        segments_temp_features.loc[i,ky] = val

segments_temp_features.dropna(inplace=True)
segments_temp_features.to_csv("data_features.csv")
# %%
