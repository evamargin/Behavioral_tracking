
import numpy as np
from scipy import signal
import statsmodels.api as sm
from scipy.signal import butter, lfilter, filtfilt
from scipy.interpolate import interp1d
import emd # check if you have the emd package installed https://emd.readthedocs.io/en/stable/api.html

# this_pitch: your pitch in degrees
  
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def moving_mean_filter(a, window):
    b = np.copy(a)
    mean_data = np.nanmean(rolling_window(a, window), -1)
    
    b[window-1:] = a[window-1:] - mean_data
    return b



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    isnan = np.isnan(signal)
    if np.any(isnan):
        # Create an interpolation function ignoring NaNs
        valid_idx = ~isnan
        interp_func = interp1d(np.flatnonzero(valid_idx), signal[valid_idx], kind='linear', bounds_error=False, fill_value="extrapolate")
        inter_res = interp_func(np.arange(len(signal)))  # Replace NaNs with interpolated values
        signal[isnan] = inter_res[isnan]    

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    
    # Padding the signal to mitigate edge effects
    padlen = 3 * max(len(b), len(a))  # Common choice for padding length
    padded_data = np.pad(signal, padlen, mode='edge')
    
    # Filtering the padded signal
    filtered_padded_data = filtfilt(b, a, padded_data, method='gust')
    
    # Removing the padding
    filtered_data = filtered_padded_data[padlen:-padlen]
    
    return filtered_data


def interpolate_func(signal):
    copy_signal = signal.copy()
    isnan = np.isnan(signal)
    if np.any(isnan):
        # Create an interpolation function ignoring NaNs
        valid_idx = ~isnan
        interp_func = interp1d(np.flatnonzero(valid_idx), signal[valid_idx], kind='linear', bounds_error=False, fill_value="extrapolate")
        inter_res = interp_func(np.arange(len(signal)))  # Replace NaNs with interpolated values
        copy_signal[isnan] = inter_res[isnan]  
    return copy_signal  


filtered_pitch = interpolate_func(this_pitch)
filtered_pitch = np.unwrap(filtered_pitch, period=360)
filtered_pitch = moving_mean_filter(filtered_pitch, 10)
    
filtered_pitch = butter_bandpass_filter(filtered_pitch, 5, 15, this_session.freq, 2)
pitch_phase, pitch_freq, pitch_power = emd.spectra.frequency_transform(filtered_pitch, this_session.freq, 'hilbert',smooth_freq=9, smooth_phase=5)
