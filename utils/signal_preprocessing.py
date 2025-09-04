"""
Signal preprocessing utilities for EEG data analysis.
This module provides functions for filtering, artifact removal, and signal conditioning.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfiltfilt
from scipy.signal.windows import get_window
from scipy.signal import hilbert
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

# Registry for preprocessing methods
PREPROCESSING_METHODS = {}

def preprocessing_method(name, description="", category="General", parameters=None):
    """
    Decorator to register a function as a preprocessing method.
    
    Parameters:
    -----------
    name : str
        Display name for the method
    description : str
        Description of what the method does
    category : str
        Category for grouping methods
    parameters : dict
        Parameter definitions with types and constraints
    """
    def decorator(func):
        PREPROCESSING_METHODS[func.__name__] = {
            'name': name,
            'function': func,
            'description': description,
            'category': category,
            'parameters': parameters or {},
            'function_name': func.__name__
        }
        return func
    return decorator

def get_preprocessing_methods():
    """Get all registered preprocessing methods."""
    return PREPROCESSING_METHODS

def get_mosaic_df(data, Mosaic_groups):
    """
    Convert a list of dictionaries to a pandas DataFrame.
    
    Parameters:
    -----------
    data : list of dict
        List of dictionaries with keys as column names
    Mosaic_list : list
        List of column names to include in the DataFrame
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with specified columns
    """
    output_dict = {}
    key_list = list(Mosaic_groups.keys())

    for key in list(Mosaic_groups.keys()):
        Mosaic_list = Mosaic_groups[key]
        output_df = pd.DataFrame()    
        output_df['time'] = data['time']
        for i in Mosaic_list:
            interest_pin = "pin_" + str(i[0])
            reference_pin = "pin_" + str(i[1])
            output_df[interest_pin + "-" + reference_pin] = data[interest_pin] - data[reference_pin]
        output_dict[key] = output_df
        
    
    return output_dict

def get_sampling_rate(data):
    """
    Estimate sampling rate from time column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with 'time' column
        
    Returns:
    --------
    fs : float
        Estimated sampling rate in Hz
    """
    if 'time' not in data.columns:
        raise ValueError("DataFrame must contain 'time' column")
    
    time_diff = np.diff(data['time'])
    median_dt = np.median(time_diff)
    fs = 1.0 / median_dt
    
    print(f"Estimated sampling rate: {fs:.1f} Hz")
    return fs

@preprocessing_method(
    name="Resample",
    description="Must do resampling before any preprocessing. Normal EEG sampling rate is 256Hz or 512Hz.",
    category="General",
    parameters={
        'target_rate': {
            'type': 'float',
            'default': 256,
            'min': 1,
            'max': 2000,
            'label': 'Target Sampling Rate (Hz)',
            'description': 'The desired sampling rate after resampling'
        }
    }
)
def resample_dataframe(data, target_rate, original_rate=None):
    """
    Resample a DataFrame to a different sampling rate.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel data
    target_rate : float
        Target sampling rate in Hz
    original_rate : float, optional
        Original sampling rate in Hz (if None, estimated from data)
        
    Returns:
    --------
    resampled_df : pandas.DataFrame
        Resampled DataFrame with updated time vector
    """
    
    # Calculate resampling factor
    if original_rate is None:
        original_rate = get_sampling_rate(data)
        
    print(f"Resampling from {original_rate:.1f} Hz to {target_rate:.1f} Hz")
        
    resample_factor = target_rate / original_rate
    new_length = int(len(data) * resample_factor)
    
    # Calculate the correct duration from the original time vector
    original_duration = data['time'].iloc[-1] - data['time'].iloc[0]
    
    # Create new time vector that maintains the correct duration and sampling rate
    # The new time vector should have the same start time and duration, but with target_rate spacing
    new_time = np.linspace(data['time'].iloc[0], data['time'].iloc[-1], new_length)
    
    # Verify the new sampling rate matches the target
    if new_length > 1:
        actual_new_rate = 1.0 / np.median(np.diff(new_time))
        print(f"Actual new sampling rate: {actual_new_rate:.1f} Hz (target: {target_rate:.1f} Hz)")
    
    # Resample each channel
    resampled_data = {'time': new_time}
    
    for column in tqdm(data.columns, desc="Resampling channels"):
        if column != 'time':
            # Use scipy's resample function
            resampled_channel = signal.resample(data[column].values, new_length)
            resampled_data[column] = resampled_channel
    
    print(f"Resampling completed: {len(data)} → {new_length} samples")
    return pd.DataFrame(resampled_data)

@preprocessing_method(
    name="Bandpass Filter",
    description="Applies a zero-phase bandpass filter to retain frequencies within a specified range. Uses Butterworth filter design.",
    category="Filtering",
    parameters={
        'lowcut': {
            'type': 'float', 
            'default': 0.5, 
            'min': 0.1, 
            'max': 1000, 
            'label': 'Low cutoff (Hz)',
            'description': 'Lower frequency bound for the bandpass filter'
        },
        'highcut': {
            'type': 'float', 
            'default': 100, 
            'min': 1, 
            'max': 1000, 
            'label': 'High cutoff (Hz)',
            'description': 'Upper frequency bound for the bandpass filter'
        },
        'order': {
            'type': 'int', 
            'default': 4, 
            'min': 1, 
            'max': 10, 
            'label': 'Filter order',
            'description': 'Order of the Butterworth filter (higher = steeper rolloff)'
        }
    }
)
def bandpass_filter(data, lowcut, highcut, fs=None, order=4):
    """
    Apply zero-phase bandpass filter to EEG data using Butterworth filter.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    order : int
        Filter order

    Returns:
    --------
    filtered_data : pandas.DataFrame
        Filtered data with same structure
    """
    if fs is None:
        fs = get_sampling_rate(data)

    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(order, [low, high], btype='bandpass', output='sos')

    filtered_data = data.copy()

    for column in data.columns:
        if column != 'time':
            filtered_data[column] = sosfiltfilt(sos, data[column].values)  # <-- 변경!

    print(f"Applied zero-phase bandpass filter: {lowcut}-{highcut} Hz")
    return filtered_data

@preprocessing_method(
    name="Lowpass Filter",
    description="Applies a lowpass filter to remove high-frequency noise and smooth the signal.",
    category="Filtering",
    parameters={
        'cutoff': {
            'type': 'float', 
            'default': 100, 
            'min': 1, 
            'max': 1000, 
            'label': 'Cutoff frequency (Hz)',
            'description': 'Frequency above which signals are attenuated'
        },
        'order': {
            'type': 'int', 
            'default': 4, 
            'min': 1, 
            'max': 10, 
            'label': 'Filter order',
            'description': 'Order of the Butterworth filter'
        }
    }
)
def lowpass_filter(data, cutoff, fs=None, order=4):
    """
    Apply lowpass filter to EEG data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    cutoff : float
        Cutoff frequency (Hz)
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    order : int
        Filter order
        
    Returns:
    --------
    filtered_data : pandas.DataFrame
        Filtered data
    """
    if fs is None:
        fs = get_sampling_rate(data)
    
    # Nyquist frequency
    nyquist = fs / 2
    
    # Normalize frequency
    normal_cutoff = cutoff / nyquist
    
    # Design filter
    sos = butter(order, normal_cutoff, btype='lowpass', output='sos')
    
    # Apply filter to each channel (except time)
    filtered_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            filtered_data[column] = sosfiltfilt(sos, data[column].values)
    
    print(f"Applied zero-phase lowpass filter: {cutoff} Hz")
    return filtered_data

@preprocessing_method(
    name="Highpass Filter",
    description="Applies a highpass filter to remove low-frequency drift and baseline wander.",
    category="Filtering",
    parameters={
        'cutoff': {
            'type': 'float', 
            'default': 0.5, 
            'min': 0.1, 
            'max': 100, 
            'label': 'Cutoff frequency (Hz)',
            'description': 'Frequency below which signals are attenuated'
        },
        'order': {
            'type': 'int', 
            'default': 4, 
            'min': 1, 
            'max': 10, 
            'label': 'Filter order',
            'description': 'Order of the Butterworth filter'
        }
    }
)
def highpass_filter(data, cutoff, fs=None, order=4):
    """
    Apply highpass filter to EEG data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    cutoff : float
        Cutoff frequency (Hz)
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    order : int
        Filter order
        
    Returns:
    --------
    filtered_data : pandas.DataFrame
        Filtered data
    """
    if fs is None:
        fs = get_sampling_rate(data)
    
    # Nyquist frequency
    nyquist = fs / 2
    
    # Normalize frequency
    normal_cutoff = cutoff / nyquist
    
    # Design filter
    sos = butter(order, normal_cutoff, btype='highpass', output='sos')
    
    # Apply filter to each channel (except time)
    filtered_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            filtered_data[column] = sosfiltfilt(sos, data[column].values)
    
    print(f"Applied zero-phase highpass filter: {cutoff} Hz")
    return filtered_data

@preprocessing_method(
    name="Notch Filter",
    description="Removes specific frequency interference (e.g., 50/60 Hz powerline noise) using a notch filter.",
    category="Filtering",
    parameters={
        'notch_freq': {
            'type': 'float', 
            'default': 50, 
            'min': 1, 
            'max': 1000, 
            'label': 'Notch frequency (Hz)',
            'description': 'Frequency to be removed (e.g., 50 Hz or 60 Hz for powerline noise)'
        },
        'quality_factor': {
            'type': 'float', 
            'default': 30, 
            'min': 1, 
            'max': 100, 
            'label': 'Quality factor',
            'description': 'Controls the width of the notch (higher = narrower)'
        }
    }
)
def notch_filter(data, notch_freq=50, quality_factor=30, fs=None):

    if fs is None:
        fs = get_sampling_rate(data)
        
    b, a = iirnotch(notch_freq, quality_factor, fs)

    filtered_data = data.copy()
    for column in data.columns:
        if column != 'time':
            filtered_data[column] = filtfilt(b, a, data[column].values)

    print(f"Applied notch filter: {notch_freq} Hz (Q={quality_factor})")
    return filtered_data

@preprocessing_method(
    name="Remove DC Offset",
    description="Removes the DC (direct current) offset from the signal by subtracting the mean value of each channel.",
    category="Baseline Correction",
    parameters={}
)
def remove_dc_offset(data):
    """
    Remove DC offset (mean) from each channel.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
        
    Returns:
    --------
    filtered_data : pandas.DataFrame
        Data with DC offset removed
    """
    filtered_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            filtered_data[column] = data[column] - data[column].mean()
    
    print("Removed DC offset from all channels")
    return filtered_data

@preprocessing_method(
    name="Z-Score Normalization",
    description="Normalizes the signal using z-score normalization (zero mean, unit variance) for each channel.",
    category="Normalization",
    parameters={}
)
def zscore_normalize(data):
    """
    Z-score normalize each channel (mean=0, std=1).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
        
    Returns:
    --------
    normalized_data : pandas.DataFrame
        Z-score normalized data
    """
    normalized_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            normalized_data[column] = (data[column] - data[column].mean()) / data[column].std()
    
    print("Applied Z-score normalization to all channels")
    return normalized_data

@preprocessing_method(
    name="Remove Artifacts (IQR)",
    description="Removes artifacts using the interquartile range (IQR) method. Detects outliers and optionally interpolates them.",
    category="Artifact Removal",
    parameters={
        'threshold': {
            'type': 'float', 
            'default': 3, 
            'min': 1, 
            'max': 10, 
            'label': 'IQR threshold',
            'description': 'Multiplier for IQR to define outlier boundaries'
        },
        'interpolate_immediately': {
            'type': 'bool', 
            'default': True, 
            'label': 'Interpolate immediately',
            'description': 'Whether to interpolate detected artifacts immediately'
        }
    }
)
def remove_artifacts_iqr(data, threshold=3, interpolate_immediately=True):
    """
    Remove artifacts using IQR-based outlier detection.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    threshold : float
        Number of IQRs beyond which data is considered artifact
    interpolate_immediately : bool
        Whether to interpolate artifacts immediately after masking
        
    Returns:
    --------
    cleaned_data : pandas.DataFrame
        Data with artifacts set to NaN and optionally interpolated
    """
    cleaned_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Set outliers to NaN
            mask = (data[column] < lower_bound) | (data[column] > upper_bound)
            cleaned_data.loc[mask, column] = np.nan
            
            artifacts_removed = mask.sum()
            if artifacts_removed > 0:
                print(f"Removed {artifacts_removed} artifacts from {column} ({artifacts_removed/len(data)*100:.2f}%)")
    
    # Optionally interpolate artifacts immediately
    if interpolate_immediately:
        print("Interpolating artifacts immediately...")
        cleaned_data = interpolate_artifacts(cleaned_data)
    
    return cleaned_data

def interpolate_artifacts(data, method='linear'):
    """
    Interpolate NaN values (artifacts) in the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns (may contain NaN)
    method : str
        Interpolation method ('linear', 'cubic', 'spline')
        
    Returns:
    --------
    interpolated_data : pandas.DataFrame
        Data with artifacts interpolated
    """
    interpolated_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            nan_count = data[column].isna().sum()
            if nan_count > 0:
                interpolated_data[column] = data[column].interpolate(method=method)
                print(f"Interpolated {nan_count} values in {column}")
    
    return interpolated_data

@preprocessing_method(
    name="Common Average Reference",
    description="Applies common average reference (CAR) to reduce common noise across all channels.",
    category="Re-referencing",
    parameters={}
)
def apply_common_average_reference(data):
    """
    Apply common average reference to EEG data.
    This removes the average signal across all channels from each channel,
    reducing common artifacts and noise.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
        
    Returns:
    --------
    car_data : pandas.DataFrame
        Data with common average reference applied
    """
    car_data = data.copy()
    
    # Calculate the average signal across all channels (excluding time)
    signal_columns = [col for col in data.columns if col != 'time']
    average_signal = data[signal_columns].mean(axis=1)
    
    # Subtract the average from each channel
    for column in signal_columns:
        car_data[column] = data[column] - average_signal
    
    print("Applied common average reference")
    return car_data


@preprocessing_method(
    name="Moving Average Smoothing",
    description="Applies a moving average smoothing filter to reduce high-frequency noise.",
    category="Smoothing",
    parameters={
        'window_size': {
            'type': 'int', 
            'default': 5, 
            'min': 3, 
            'max': 51, 
            'label': 'Window size (samples)',
            'description': 'Number of samples to average (should be odd for symmetry)'
        }
    }
)
def moving_average_smoothing(data, window_size=5):
    """
    Apply moving average smoothing to EEG data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    window_size : int
        Size of the moving average window
    
    Returns:
    --------
    smoothed_data : pandas.DataFrame
        Smoothed data with same structure
    """
    import numpy as np
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        print(f"Adjusted window size to {window_size} (made odd)")
    
    # Create smoothing kernel
    kernel = np.ones(window_size) / window_size
    
    smoothed_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            # Apply convolution with 'same' mode to preserve signal length
            smoothed_data[column] = np.convolve(data[column].values, kernel, mode='same')
    
    print(f"Applied moving average smoothing with window size {window_size}")
    return smoothed_data
    """
    Apply Common Average Reference (CAR) to reduce common noise.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
        
    Returns:
    --------
    car_data : pandas.DataFrame
        Data with CAR applied
    """
    car_data = data.copy()
    
    # Get channel columns (exclude time)
    channel_columns = [col for col in data.columns if col != 'time']
    
    # Calculate common average
    common_average = data[channel_columns].mean(axis=1)
    
    # Subtract common average from each channel
    for column in channel_columns:
        car_data[column] = data[column] - common_average
    
    print(f"Applied Common Average Reference across {len(channel_columns)} channels")
    return car_data

def interpolate_nans_polynomial(arr, order=3):
    """
    Interpolate NaN values using polynomial interpolation of specified order.

    Parameters:
    -----------
    arr : np.ndarray
        1D array with NaNs
    order : int
        Order of the polynomial to fit

    Returns:
    --------
    np.ndarray
        Interpolated array
    """
    arr = np.asarray(arr, dtype=float)
    x = np.arange(len(arr))
    known_x = x[~np.isnan(arr)]
    known_y = arr[~np.isnan(arr)]

    coeffs = np.polyfit(known_x, known_y, order)
    poly = np.poly1d(coeffs)
    arr[np.isnan(arr)] = poly(x[np.isnan(arr)])
    return arr

def remove_artifacts_chunk(sig, fs, env_threshold=5, kurtosis_threshold=3):
    output = sig.copy()
    # 힐버트 앰플리튜드 기반 절대 임계치
    env = np.abs(hilbert(sig))
    mad = np.median(np.abs(env - np.median(env)))
    thr_env = np.median(env) + env_threshold * mad
    mask_env = env < thr_env
    # kurtosis 기반 절대 임계치
    kur = kurtosis(sig)
    mad_k = np.median(np.abs(kur - np.median(kur)))
    thr_k = np.median(kur) + kurtosis_threshold * mad_k
    mask_k = np.ones_like(sig, dtype=bool)
    if kur > thr_k:
        mask_k[:] = False

    combined_mask = mask_env & mask_k
    artifacts_removed = np.sum(~combined_mask)
    # print(f"Masked {artifacts_removed} artifact samples in {artifacts_removed/len(sig)*100:.5f}%)")

    output[~combined_mask] = np.nan

    return output

@preprocessing_method(
    name="Remove Artifacts (Hilbert-Kurtosis)",
    description="Advanced artifact removal using Hilbert transform envelope analysis and kurtosis-based detection.",
    category="Artifact Removal",
    parameters={
        'env_threshold': {
            'type': 'float', 
            'default': 5, 
            'min': 1, 
            'max': 20, 
            'label': 'Envelope threshold',
            'description': 'Threshold for envelope-based artifact detection'
        },
        'kurtosis_threshold': {
            'type': 'float', 
            'default': 3, 
            'min': 1, 
            'max': 10, 
            'label': 'Kurtosis threshold',
            'description': 'Threshold for kurtosis-based artifact detection'
        },
        'interpolate': {
            'type': 'bool', 
            'default': True, 
            'label': 'Interpolate artifacts',
            'description': 'Whether to interpolate detected artifacts'
        }
    }
)
def remove_artifacts_hilbert_kurtosis(data, fs=None, env_threshold=5, kurtosis_threshold=3, 
                                    window_size=2.0, step_size=1.0, interpolate=True):
    """
    Remove artifacts using Hilbert envelope and kurtosis-based detection with sliding window.
    Advanced method for EEG artifact removal with chunk-based processing.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    env_threshold : float
        Threshold multiplier for Hilbert envelope (MAD-based)
    kurtosis_threshold : float
        Threshold multiplier for kurtosis detection (MAD-based)
    window_size : float
        Window size in seconds for chunk processing
    step_size : float
        Step size in seconds for sliding window
    interpolate : bool
        Whether to interpolate artifacts immediately after masking
        
    Returns:
    --------
    cleaned_data : pandas.DataFrame
        Data with artifacts removed and optionally interpolated
    """
    if fs is None:
        fs = get_sampling_rate(data)
    
    file_len = data.shape[0]
    file_len_s = data.shape[0] /fs

    window_s = int(window_size)
    step_s = int(step_size)
    
    cleaned_data = data.copy()
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    artifact_mask = np.zeros(file_len, dtype=bool)
    total_artifacts = 0
    
    for column in data.columns:
        if column != 'time':
            sig = data[column].values
            cleaned_sig = sig.copy()
            for start_sample in tqdm(np.arange(0, file_len-window_samples+step_samples, step_samples)):
                s_idx = start_sample
                e_idx = start_sample + window_samples
                chunk = sig[s_idx:e_idx]

                cleaned_chunk = remove_artifacts_chunk(chunk, fs, env_threshold=env_threshold, kurtosis_threshold=kurtosis_threshold)
                # Process chunk

                if interpolate:
                    # Interpolate NaNs immediately in the chunk
                    cleaned_chunk = interpolate_nans_polynomial(cleaned_chunk, order=1)
                
                # Accumulate artifact detections (any NaN in any window marks as artifact)
                chunk_artifact_mask = np.isnan(cleaned_chunk)
                artifact_mask[s_idx:e_idx] |= chunk_artifact_mask
                
                # Count artifacts in this chunk
                chunk_artifacts = np.sum(chunk_artifact_mask)
                total_artifacts += chunk_artifacts

                cleaned_sig[s_idx:e_idx] = cleaned_chunk
            
        
            if total_artifacts > 0:
                print(f"Processed {column} with sliding window ({window_size}s window, {step_size}s step)")
                print(f"  Total artifact samples marked: {total_artifacts} ({total_artifacts/len(sig)*100:.2f}%)")
            
            cleaned_data[column] = cleaned_sig
    
    return cleaned_data

def remove_artifacts_adaptive_threshold(data, fs=None, window_size=1.0, threshold_factor=3, interpolate=True):
    """
    Remove artifacts using adaptive threshold based on local signal statistics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with time and channel columns
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    window_size : float
        Window size in seconds for local statistics
    threshold_factor : float
        Threshold multiplier for artifact detection
    interpolate_immediately : bool
        Whether to interpolate artifacts immediately after masking
        
    Returns:
    --------
    cleaned_data : pandas.DataFrame
        Data with artifacts removed and optionally interpolated
    """
    if fs is None:
        fs = get_sampling_rate(data)
    
    window_samples = int(window_size * fs)
    cleaned_data = data.copy()
    
    for column in data.columns:
        if column != 'time':
            sig = data[column].values
            mask = np.ones_like(sig, dtype=bool)
            
            # Sliding window for adaptive thresholding
            for i in range(0, len(sig), window_samples // 2):
                start_idx = max(0, i - window_samples // 2)
                end_idx = min(len(sig), i + window_samples // 2)
                window = sig[start_idx:end_idx]
                
                # Local statistics
                local_median = np.median(window)
                local_mad = np.median(np.abs(window - local_median))
                
                # Adaptive threshold
                threshold_upper = local_median + threshold_factor * local_mad
                threshold_lower = local_median - threshold_factor * local_mad
                
                # Apply to current window
                window_start = max(0, i - window_samples // 4)
                window_end = min(len(sig), i + window_samples // 4)
                
                window_mask = (sig[window_start:window_end] >= threshold_lower) & \
                             (sig[window_start:window_end] <= threshold_upper)
                
                mask[window_start:window_end] &= window_mask
            
            # Apply mask
            cleaned_sig = sig.copy()
            cleaned_sig[~mask] = np.nan
            cleaned_data[column] = cleaned_sig
            
            artifacts_removed = np.sum(~mask)
            if artifacts_removed > 0:
                print(f"Removed {artifacts_removed} artifacts from {column} using adaptive threshold ({artifacts_removed/len(sig)*100:.2f}%)")
    
    # Optionally interpolate artifacts immediately
    if interpolate:
        print("Interpolating artifacts immediately...")
        cleaned_data = interpolate_artifacts(cleaned_data)
    
    return cleaned_data

def preprocess_eeg_data(data, 
                       fs=None,
                       bandpass_range=None,
                       lowpass_cutoff=None,
                       highpass_cutoff=None,
                       notch_freq=None,
                       remove_dc=False,
                       apply_car=False,
                       remove_artifacts=False,
                       artifact_methods=['hilbert_kurtosis'],  # 'iqr', 'hilbert_kurtosis', 'adaptive'
                       artifact_threshold=3,
                       env_threshold=5,
                       kurtosis_threshold=3,
                       window_size=2.0,
                       step_size=1.0,
                       interpolate=True,
                       interpolate_immediately=False,
                       normalize=False,
                       save_npz=False,
                       original_npz_path=None,
                       force_overwrite=False):
    """
    Complete EEG preprocessing pipeline.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw EEG data with time and channel columns
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    bandpass_range : tuple, optional
        (low_freq, high_freq) for bandpass filter
    lowpass_cutoff : float, optional
        Lowpass filter cutoff frequency
    highpass_cutoff : float, optional
        Highpass filter cutoff frequency
    notch_freq : float or list
        Frequency(ies) to notch out (50 or 60 Hz for power line)
    remove_dc : bool
        Whether to remove DC offset
    apply_car : bool
        Whether to apply Common Average Reference
    remove_artifacts : bool
        Whether to remove artifacts
    artifact_methods : list
        Methods for artifact removal: 'iqr', 'hilbert_kurtosis', 'adaptive'
    artifact_threshold : float
        IQR threshold for artifact detection (for 'iqr' method)
    env_threshold : float
        Hilbert envelope threshold (for 'hilbert_kurtosis' method)
    kurtosis_threshold : float
        Kurtosis threshold (for 'hilbert_kurtosis' method)
    window_size : float
        Window size in seconds for sliding window (for 'hilbert_kurtosis' method)
    step_size : float
        Step size in seconds for sliding window (for 'hilbert_kurtosis' method)
    interpolate : bool
        Whether to interpolate artifacts at the end of preprocessing
    interpolate_immediately : bool
        Whether to interpolate artifacts immediately after each removal method
    normalize : bool
        Whether to apply Z-score normalization
    save_npz : bool
        Whether to save processed data as NPZ file
    original_npz_path : str, optional
        Path to original NPZ file (required if save_npz=True)
    force_overwrite : bool
        Whether to force overwrite existing processed files
        
    Returns:
    --------
    processed_data : pandas.DataFrame
        Preprocessed EEG data
    output_path : str, optional
        Path to saved NPZ file (if save_npz=True)
    """
    print("🧠 Starting EEG preprocessing pipeline...")
    processed_data = data.copy()
    
    # Estimate sampling rate if not provided
    if fs is None:
        fs = get_sampling_rate(processed_data)
    
    # Check if processed file already exists and adjust save_npz flag
    if save_npz and original_npz_path and not force_overwrite:
        from pathlib import Path
        original_path = Path(original_npz_path)
        processed_path = original_path.parent / f"{original_path.stem}_processed{original_path.suffix}"
        
        if processed_path.exists():
            print(f"⚠️  Processed file already exists: {processed_path}")
            print("   Setting save_npz=False to avoid overwriting existing data.")
            print("   Use force_overwrite=True to overwrite existing files.")
            save_npz = False
    
    # If save_npz is False and processed file exists, load and return it instead of processing
    if not save_npz and original_npz_path:
        from pathlib import Path
        original_path = Path(original_npz_path)
        processed_path = original_path.parent / f"{original_path.stem}_processed{original_path.suffix}"
        
        if processed_path.exists():
            print(f"📂 Loading existing processed file: {processed_path}")
            print("   Skipping preprocessing pipeline.")
            loaded_data, loaded_metadata = load_processed_data_npz(processed_path)
            return loaded_data, processed_path
    
    # 1. Remove DC offset
    if remove_dc:
        processed_data = remove_dc_offset(processed_data)
    
    # 2. Apply filters
    if bandpass_range is not None:
        low_freq, high_freq = bandpass_range
        processed_data = bandpass_filter(processed_data, low_freq, high_freq, fs)
    else:
        if highpass_cutoff is not None:
            processed_data = highpass_filter(processed_data, highpass_cutoff, fs)
        if lowpass_cutoff is not None:
            processed_data = lowpass_filter(processed_data, lowpass_cutoff, fs)
    
    # 3. Notch filter(s)
    if notch_freq is not None:
        if isinstance(notch_freq, (list, tuple)):
            for freq in notch_freq:
                processed_data = notch_filter(processed_data, freq, fs=fs)
        else:
            processed_data = notch_filter(processed_data, notch_freq, fs=fs)
    
    # 4. Common Average Reference
    if apply_car:
        processed_data = apply_common_average_reference(processed_data)
    
    # 5. Artifact removal
    if remove_artifacts:
        for artifact_method in artifact_methods:
            if artifact_method == 'iqr':
                processed_data = remove_artifacts_iqr(processed_data, artifact_threshold, interpolate_immediately)
            elif artifact_method == 'hilbert_kurtosis':
                processed_data = remove_artifacts_hilbert_kurtosis(
                    processed_data, fs, env_threshold, kurtosis_threshold, window_size, step_size, interpolate
                )
            elif artifact_method == 'adaptive':
                processed_data = remove_artifacts_adaptive_threshold(
                    processed_data, fs, window_size=window_size, threshold_factor=artifact_threshold, 
                    interpolate=interpolate
                )
            else:
                print(f"Warning: Unknown artifact method '{artifact_method}', using IQR method")
                processed_data = remove_artifacts_iqr(processed_data, artifact_threshold, interpolate_immediately)
    
    # 6. Interpolation (only if not done immediately after each artifact removal)
    if interpolate and not interpolate_immediately:
        processed_data = interpolate_artifacts(processed_data)
    
    # 7. Normalization
    if normalize:
        processed_data = zscore_normalize(processed_data)
    
    print("✅ EEG preprocessing pipeline completed!")

    # 8. Save processed data if requested
    if save_npz and original_npz_path:
        # Collect preprocessing parameters
        preprocessing_info = {
            'fs': fs,
            'bandpass_range': bandpass_range,
            'lowpass_cutoff': lowpass_cutoff,
            'highpass_cutoff': highpass_cutoff,
            'notch_freq': notch_freq,
            'remove_dc': remove_dc,
            'apply_car': apply_car,
            'remove_artifacts': remove_artifacts,
            'artifact_methods': artifact_methods,
            'artifact_threshold': artifact_threshold,
            'env_threshold': env_threshold,
            'kurtosis_threshold': kurtosis_threshold,
            'window_size': window_size,
            'step_size': step_size,
            'interpolate': interpolate,
            'normalize': normalize
        }
        
        output_path = save_processed_data_npz_force(
            processed_data, original_npz_path, fs, preprocessing_info, overwrite=force_overwrite
        )
        return processed_data, output_path
    print("test")
    
    return processed_data, processed_path
# Preset configurations for common EEG analysis
def preprocess_for_general_analysis(data, fs=None, artifact_method='hilbert_kurtosis', 
                                   window_size=2.0, step_size=1.0, interpolate_immediately=True,
                                   save_npz=False, original_npz_path=None, force_overwrite=False):
    """
    Preset for general EEG analysis.
    - Bandpass: 0.5-50 Hz
    - Notch: 50 Hz
    - Advanced artifact removal with sliding window
    - Immediate interpolation of artifacts (default)
    """
    return preprocess_eeg_data(
        data, 
        fs=fs,
        bandpass_range=(0.5, 50),
        notch_freq=50,
        remove_dc=True,
        apply_car=True,
        remove_artifacts=True,
        artifact_methods=[artifact_method],
        window_size=window_size,
        step_size=step_size,
        interpolate=True,
        interpolate_immediately=interpolate_immediately,
        normalize=False,
        save_npz=save_npz,
        original_npz_path=original_npz_path,
        force_overwrite=force_overwrite
    )

def preprocess_for_seizure_detection(data, fs=None, artifact_method='hilbert_kurtosis', 
                                    window_size=2.0, step_size=1.0, interpolate_immediately=True,
                                    save_npz=False, original_npz_path=None, force_overwrite=False):
    """
    Preset for seizure detection.
    - Bandpass: 1-70 Hz
    - Notch: 50 Hz
    - Advanced artifact removal with smaller window for sensitivity
    - Immediate interpolation of artifacts (default)
    """
    return preprocess_eeg_data(
        data,
        fs=fs,
        bandpass_range=(1, 70),
        notch_freq=50,
        remove_dc=True,
        apply_car=False,  # Keep original amplitudes for seizure detection
        remove_artifacts=True,
        artifact_methods=[artifact_method],
        env_threshold=4,  # More sensitive for seizure detection
        kurtosis_threshold=2.5,  # More aggressive
        window_size=window_size,
        step_size=step_size,
        interpolate=True,
        interpolate_immediately=interpolate_immediately,
        normalize=False,
        save_npz=save_npz,
        original_npz_path=original_npz_path,
        force_overwrite=force_overwrite
    )

def preprocess_for_spectral_analysis(data, fs=None, save_npz=False, original_npz_path=None, force_overwrite=False):
    """
    Preset for spectral/frequency analysis.
    - Highpass: 0.1 Hz (preserve very low frequencies)
    - Lowpass: 100 Hz
    - Notch: 50 Hz
    - Minimal artifact removal to preserve frequency content
    """
    return preprocess_eeg_data(
        data,
        fs=fs,
        highpass_cutoff=0.1,
        lowpass_cutoff=100,
        notch_freq=50,
        remove_dc=True,
        apply_car=False,
        remove_artifacts=False,  # Preserve signal for frequency analysis
        interpolate=False,
        normalize=False,
        save_npz=save_npz,
        original_npz_path=original_npz_path,
        force_overwrite=force_overwrite
    )

def save_processed_data_npz(processed_data, original_npz_path, fs=None, preprocessing_info=None):
    """
    Save processed EEG data to NPZ format with "_processed" suffix.
    
    Parameters:
    -----------
    processed_data : pandas.DataFrame
        Processed EEG data with time and channel columns
    original_npz_path : str
        Path to the original NPZ file
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    preprocessing_info : dict, optional
        Dictionary containing preprocessing parameters used
        
    Returns:
    --------
    output_path : str
        Path to the saved processed NPZ file
    """
    import os
    from pathlib import Path
    
    # Create output path with "_processed" suffix
    original_path = Path(original_npz_path)
    output_path = original_path.parent / f"{original_path.stem}_processed{original_path.suffix}"
    
    # Check if processed file already exists
    if output_path.exists():
        print(f"⚠️  Processed file already exists: {output_path}")
        print("   Skipping save operation to avoid overwriting existing data.")
        return str(output_path)
    
    # Estimate sampling rate if not provided
    if fs is None:
        fs = get_sampling_rate(processed_data)
    
    # Prepare data for saving
    save_dict = {
        'time': processed_data['time'].values,
        'sampling_rate': fs
    }
    
    # Add channel data
    channel_columns = [col for col in processed_data.columns if col != 'time']
    for channel in channel_columns:
        save_dict[channel] = processed_data[channel].values
    
    # Add preprocessing information
    if preprocessing_info is not None:
        save_dict['preprocessing_info'] = preprocessing_info
    else:
        save_dict['preprocessing_info'] = {'status': 'processed'}
    
    # Add metadata
    save_dict['metadata'] = {
        'channels': channel_columns,
        'n_samples': len(processed_data),
        'duration_seconds': processed_data['time'].iloc[-1] - processed_data['time'].iloc[0],
        'processed_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to NPZ file
    np.savez_compressed(output_path, **save_dict)
    
    print(f"✅ Processed data saved to: {output_path}")
    print(f"   - Channels: {len(channel_columns)}")
    print(f"   - Samples: {len(processed_data):,}")
    print(f"   - Duration: {save_dict['metadata']['duration_seconds']:.2f} seconds")
    print(f"   - Sampling rate: {fs:.1f} Hz")
    
    return str(output_path)

def save_processed_data_npz_force(processed_data, original_npz_path, fs=None, preprocessing_info=None, overwrite=True):
    """
    Save processed EEG data to NPZ format with "_processed" suffix, with option to force overwrite.
    
    Parameters:
    -----------
    processed_data : pandas.DataFrame
        Processed EEG data with time and channel columns
    original_npz_path : str
        Path to the original NPZ file
    fs : float, optional
        Sampling rate (Hz). If None, estimated from data
    preprocessing_info : dict, optional
        Dictionary containing preprocessing parameters used
    overwrite : bool
        Whether to overwrite existing processed file
        
    Returns:
    --------
    output_path : str
        Path to the saved processed NPZ file
    """
    import os
    from pathlib import Path
    
    # Create output path with "_processed" suffix
    original_path = Path(original_npz_path)
    output_path = original_path.parent / f"{original_path.stem}_processed{original_path.suffix}"
    
    # Check if processed file already exists
    if output_path.exists() and not overwrite:
        print(f"⚠️  Processed file already exists: {output_path}")
        print("   Use overwrite=True to force overwrite existing data.")
        return str(output_path)
    elif output_path.exists() and overwrite:
        print(f"🔄 Overwriting existing processed file: {output_path}")
    
    # Estimate sampling rate if not provided
    if fs is None:
        fs = get_sampling_rate(processed_data)
    
    # Prepare data for saving
    save_dict = {
        'time': processed_data['time'].values,
        'sampling_rate': fs
    }
    
    # Add channel data
    channel_columns = [col for col in processed_data.columns if col != 'time']
    for channel in channel_columns:
        save_dict[channel] = processed_data[channel].values
    
    # Add preprocessing information
    if preprocessing_info is not None:
        save_dict['preprocessing_info'] = preprocessing_info
    else:
        save_dict['preprocessing_info'] = {'status': 'processed'}
    
    # Add metadata
    save_dict['metadata'] = {
        'channels': channel_columns,
        'n_samples': len(processed_data),
        'duration_seconds': processed_data['time'].iloc[-1] - processed_data['time'].iloc[0],
        'processed_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save to NPZ file
    np.savez_compressed(output_path, **save_dict)
    
    print(f"✅ Processed data saved to: {output_path}")
    print(f"   - Channels: {len(channel_columns)}")
    print(f"   - Samples: {len(processed_data):,}")
    print(f"   - Duration: {save_dict['metadata']['duration_seconds']:.2f} seconds")
    print(f"   - Sampling rate: {fs:.1f} Hz")
    
    return str(output_path)

def load_processed_data_npz(npz_path):
    """
    Load processed EEG data from NPZ file.
    
    Parameters:
    -----------
    npz_path : str
        Path to the processed NPZ file
        
    Returns:
    --------
    data : pandas.DataFrame
        Loaded EEG data
    metadata : dict
        Metadata information
    """
    # Load NPZ file
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Create DataFrame
    data_dict = {'time': npz_data['time']}
    
    # Load channel data
    metadata = npz_data['metadata'].item() if 'metadata' in npz_data else {}
    channels = metadata.get('channels', [])
    
    if not channels:
        # If no metadata, find channel columns
        channels = [key for key in npz_data.keys() 
                   if key not in ['time', 'sampling_rate', 'preprocessing_info', 'metadata']]
    
    for channel in channels:
        if channel in npz_data:
            data_dict[channel] = npz_data[channel]
    
    data = pd.DataFrame(data_dict)
    
    # Extract metadata
    full_metadata = {
        'sampling_rate': float(npz_data['sampling_rate']) if 'sampling_rate' in npz_data else None,
        'preprocessing_info': npz_data['preprocessing_info'].item() if 'preprocessing_info' in npz_data else {},
        'file_info': metadata
    }
    
    print(f"✅ Loaded processed data from: {npz_path}")
    print(f"   - Channels: {len(channels)}")
    print(f"   - Samples: {len(data):,}")
    if full_metadata['sampling_rate']:
        print(f"   - Sampling rate: {full_metadata['sampling_rate']:.1f} Hz")
    
    return data, full_metadata

if __name__ == "__main__":
    print("EEG Signal Preprocessing Module")
    print("\nAvailable functions:")
    print("- bandpass_filter(data, lowcut, highcut)")
    print("- lowpass_filter(data, cutoff)")
    print("- highpass_filter(data, cutoff)")
    print("- notch_filter(data, notch_freq)")
    print("- remove_artifacts_iqr(data, threshold)")
    print("- remove_artifacts_hilbert_kurtosis(data, env_threshold, kurtosis_threshold)")
    print("- remove_artifacts_adaptive_threshold(data, threshold_factor)")
    print("- preprocess_eeg_data(data, **kwargs)")
    print("\nPreset configurations:")
    print("- preprocess_for_general_analysis(data, artifact_method='hilbert_kurtosis')")
    print("- preprocess_for_seizure_detection(data, artifact_method='hilbert_kurtosis')")
    print("- preprocess_for_spectral_analysis(data)")
    print("\nArtifact removal methods:")
    print("- 'iqr': Traditional IQR-based outlier detection")
    print("- 'hilbert_kurtosis': Hilbert envelope + kurtosis (recommended)")
    print("- 'adaptive': Adaptive threshold based on local statistics")