"""
Data format conversion utilities for EEG preprocessing.

This module provides functions to convert between different data formats
used in the EEG analysis pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def create_minimal_metadata(data_df, sampling_rate=None):
    """
    Create minimal metadata for CSV data that doesn't have full RHD metadata.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame with 'time' column and channel columns
    sampling_rate : float, optional
        Sampling rate in Hz. If None, calculated from time column
    
    Returns:
    --------
    metadata : dict
        Minimal metadata dictionary with sampling frequency, channel names, and sample length
    """
    # Get channel names (excluding 'time' column)
    channel_columns = [col for col in data_df.columns if col != 'time']
    
    # Calculate sampling rate if not provided
    if sampling_rate is None and 'time' in data_df.columns:
        time_diff = np.diff(data_df['time'].values)
        median_dt = np.median(time_diff)
        sampling_rate = 1.0 / median_dt
    
    # Get total sample length
    n_samples = len(data_df)
    
    # Create minimal metadata structure
    metadata = {
        'frequency_parameters': {
            'amplifier_sample_rate': sampling_rate
        },
        'amplifier_channels': [
            {'native_channel_name': ch, 'custom_channel_name': ch}
            for ch in channel_columns
        ],
        'amplifier_data': None,  # Will be filled later
        't_amplifier': data_df['time'].values if 'time' in data_df.columns else np.arange(n_samples) / sampling_rate,
        'data_format': 'csv',  # Mark this as CSV-originated data
        'n_samples': n_samples
    }
    
    return metadata


def convert_rhd_result_to_dataframe(result, sampling_rate=None):
    """
    Convert RHD result dictionary to pandas DataFrame format expected by preprocessing functions.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from read_intan containing 'amplifier_data' and 'amplifier_channels'
        OR a minimal metadata dict created from CSV data
    sampling_rate : float, optional
        Sampling rate in Hz. If None, extracted from result['frequency_parameters']
    
    Returns:
    --------
    data_df : pandas.DataFrame
        DataFrame with 'time' column and channel columns
    """
    if 'amplifier_data' not in result:
        raise ValueError("Result dictionary must contain 'amplifier_data'")
    
    amplifier_data = result['amplifier_data']  # Shape: (n_channels, n_samples)
    
    # Handle time vector - use existing t_amplifier or create one
    if 't_amplifier' in result and result['t_amplifier'] is not None:
        time_vector = np.array(result['t_amplifier'])  # Time vector in seconds
    else:
        # Create time vector from sampling rate
        if sampling_rate is None:
            if 'frequency_parameters' in result:
                sampling_rate = result['frequency_parameters'].get('amplifier_sample_rate')
            
            if sampling_rate is None:
                raise ValueError("Sampling rate not found in result and not provided")
        
        n_samples = amplifier_data.shape[1] if amplifier_data.ndim > 1 else len(amplifier_data)
        time_vector = np.arange(n_samples) / sampling_rate

    if amplifier_data.ndim > 1:
        assert len(time_vector) == amplifier_data.shape[1], "Time vector length must match number of samples"
    
    # Get sampling rate
    if sampling_rate is None:
        if 'frequency_parameters' in result:
            sampling_rate = result['frequency_parameters'].get('amplifier_sample_rate')
        
        if sampling_rate is None:
            raise ValueError("Sampling rate not found in result and not provided")
    
    # Get number of samples
    if amplifier_data.ndim == 1:
        n_samples = len(amplifier_data)
        n_channels = 1
        amplifier_data = amplifier_data.reshape(1, -1)
    else:
        n_channels, n_samples = amplifier_data.shape
    
    # Create DataFrame
    data_dict = {'time': time_vector}
    
    # Add channel data
    if 'amplifier_channels' in result:
        channel_info = result['amplifier_channels']
        for i in range(min(n_channels, len(channel_info))):
            # Use the native channel name from the RHD file directly
            channel_name = channel_info[i].get('native_channel_name', f'CH{i+1}')
            data_dict[channel_name] = amplifier_data[i, :]
    else:
        # If no channel info, use generic names
        for i in range(n_channels):
            data_dict[f'CH{i+1}'] = amplifier_data[i, :]
    
    return pd.DataFrame(data_dict)


def convert_dataframe_to_rhd_result(data_df, original_result, sampling_rate=None):
    """
    Convert processed DataFrame back to RHD result format for caching.
    Handles both RHD-originated and CSV-originated data.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        Processed DataFrame with 'time' column and channel columns
    original_result : dict
        Original result dictionary to preserve metadata
    sampling_rate : float, optional
        Sampling rate in Hz. If None, estimated from time column
    
    Returns:
    --------
    updated_result : dict
        Updated result dictionary with processed amplifier_data
    """
    # Create a copy of the original result
    updated_result = original_result.copy()
    
    # Get channel columns (excluding 'time')
    channel_columns = [col for col in data_df.columns if col != 'time']
    
    # Convert DataFrame back to numpy array format
    amplifier_data = np.array([data_df[col].values for col in channel_columns])
    
    # Update the amplifier_data in the result
    updated_result['amplifier_data'] = amplifier_data
    
    # Update time vector
    if 'time' in data_df.columns:
        updated_result['t_amplifier'] = data_df['time'].values
    
    # Update sampling rate if provided or calculate from time
    if sampling_rate is not None:
        if 'frequency_parameters' not in updated_result:
            updated_result['frequency_parameters'] = {}
        updated_result['frequency_parameters']['amplifier_sample_rate'] = sampling_rate
    elif 'time' in data_df.columns and len(data_df) > 1:
        # Recalculate sampling rate from time column
        time_diff = np.diff(data_df['time'].values)
        median_dt = np.median(time_diff)
        calculated_rate = 1.0 / median_dt
        if 'frequency_parameters' not in updated_result:
            updated_result['frequency_parameters'] = {}
        updated_result['frequency_parameters']['amplifier_sample_rate'] = calculated_rate
    
    # Update channel information if needed
    if 'amplifier_channels' not in updated_result or len(updated_result['amplifier_channels']) != len(channel_columns):
        updated_result['amplifier_channels'] = [
            {'native_channel_name': ch, 'custom_channel_name': ch}
            for ch in channel_columns
        ]
    
    # Update sample count
    updated_result['n_samples'] = len(data_df)
    
    # Add preprocessing metadata
    if 'preprocessing_applied' not in updated_result:
        updated_result['preprocessing_applied'] = True
        updated_result['preprocessing_timestamp'] = pd.Timestamp.now().isoformat()
    
    return updated_result


def prepare_data_for_preprocessing(data_list):
    """
    Prepare a list of (filename, result, data_present) tuples for preprocessing.
    Converts each result to DataFrame format.
    Handles both RHD data and CSV data by creating minimal metadata for CSV.
    
    Parameters:
    -----------
    data_list : list
        List of (filename, result_dict, data_present_bool) tuples
    
    Returns:
    --------
    prepared_data : list
        List of (filename, dataframe, data_present_bool, original_result) tuples
    """
    prepared_data = []
    
    for filename, result, data_present in data_list:
        if not data_present or not result:
            prepared_data.append((filename, None, data_present, result))
            continue
        
        try:
            # Check if this is CSV data (DataFrame) or RHD data (dict)
            if isinstance(result, pd.DataFrame):
                # CSV data - create minimal metadata
                print(f"Processing CSV data: {filename}")
                
                # Create minimal metadata from DataFrame
                metadata = create_minimal_metadata(result)
                
                # Convert DataFrame columns to amplifier_data format
                channel_columns = [col for col in result.columns if col != 'time']
                amplifier_data = np.array([result[col].values for col in channel_columns])
                metadata['amplifier_data'] = amplifier_data
                
                # Use the DataFrame directly for preprocessing
                data_df = result
                prepared_data.append((filename, data_df, data_present, metadata))
                
            else:
                # RHD data - use existing conversion
                data_df = convert_rhd_result_to_dataframe(result)
                prepared_data.append((filename, data_df, data_present, result))
            
        except Exception as e:
            print(f"Warning: Could not convert {filename} to DataFrame: {e}")
            import traceback
            traceback.print_exc()
            prepared_data.append((filename, None, False, result))
    
    return prepared_data


def finalize_processed_data(processed_data_list):
    """
    Convert processed DataFrames back to RHD result format for caching.
    
    Parameters:
    -----------
    processed_data_list : list
        List of (filename, processed_dataframe, data_present_bool, original_result, [sampling_rate]) tuples
    
    Returns:
    --------
    final_data : list
        List of (filename, updated_result_dict, data_present_bool) tuples ready for caching
    """
    final_data = []
    
    for item in processed_data_list:
        # Handle both old format (4 items) and new format (5 items with sampling rate)
        if len(item) == 5:
            filename, processed_df, data_present, original_result, sampling_rate = item
        else:
            filename, processed_df, data_present, original_result = item
            sampling_rate = None
        
        if not data_present or processed_df is None:
            final_data.append((filename, original_result, data_present))
            continue
        
        try:
            # If sampling rate was updated during preprocessing, recalculate from time column
            if sampling_rate is None and processed_df is not None and 'time' in processed_df.columns:
                time_diff = np.diff(processed_df['time'])
                median_dt = np.median(time_diff)
                sampling_rate = 1.0 / median_dt
                print(f"Recalculated sampling rate for {filename}: {sampling_rate:.1f} Hz")
            
            # Convert processed DataFrame back to result format
            updated_result = convert_dataframe_to_rhd_result(processed_df, original_result, sampling_rate)
            final_data.append((filename, updated_result, data_present))
            
        except Exception as e:
            print(f"Warning: Could not convert processed {filename} back to result format: {e}")
            final_data.append((filename, original_result, data_present))
    
    return final_data
