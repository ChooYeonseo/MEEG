"""
Utility functions for reading Intan RHD files.
This module provides functions to load and process EEG data from Intan RHD files.
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


# Add the load-rhd-notebook-python directory to the Python path
current_dir = Path(__file__).parent
rhd_utils_dir = current_dir.parent / "load-rhd-notebook-python"
sys.path.insert(0, str(rhd_utils_dir))

# Import the RHD utilities
try:
    import importrhdutilities as rhd_utils
    print("Successfully imported RHD utilities")
except ImportError as e:
    print(f"Error importing RHD utilities: {e}")
    raise

def get_sample_rate(result):
    """
    Get the sample rate from the result dictionary.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from load_file()
        
    Returns:
    --------
    sample_rate : float
        Sample rate in Hz
    """
    if 'frequency_parameters' in result and 'amplifier_sample_rate' in result['frequency_parameters']:
        return int(result['frequency_parameters']['amplifier_sample_rate'])
    else:
        print("Sample rate not found in result")
        return None


def read_rhd_file(file_path):
    """
    Read a single RHD file and return the result.
    
    Parameters:
    -----------
    file_path : str
        Path to the RHD file to read
        
    Returns:
    --------
    result : dict
        Dictionary containing the loaded RHD data
    data_present : bool
        Boolean indicating if data was present in the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"RHD file not found: {file_path}")
    
    result, data_present = rhd_utils.load_file(file_path)
    
    if data_present:
        print(f"Successfully loaded data from {os.path.basename(file_path)}")
    else:
        print(f"No data found in {os.path.basename(file_path)}")
    
    return result, data_present


def read_rhd_directory(directory_path, file_pattern="*.rhd"):
    """
    Read all RHD files from a given directory.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory containing RHD files
    file_pattern : str, optional
        File pattern to match (default: "*.rhd")
        
    Returns:
    --------
    results : list
        List of tuples containing (filename, result_dict, data_present_bool)
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all RHD files in the directory
    rhd_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not rhd_files:
        print(f"No RHD files found in {directory_path}")
        return []
    
    print(f"Found {len(rhd_files)} RHD files in {directory_path}")
    
    results = []
    for file_path in sorted(rhd_files):
        try:
            result, data_present = read_rhd_file(file_path)
            filename = os.path.basename(file_path)
            results.append((filename, result, data_present))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return results


def get_channel_data(result, channel_name):
    """
    Extract data for a specific channel from the result dictionary.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from load_file()
    channel_name : str
        Name of the channel to extract
        
    Returns:
    --------
    channel_data : numpy.ndarray or None
        Channel data array, or None if channel not found
    """
    # First, try to find the channel in amplifier_channels and get the data
    if 'amplifier_channels' in result and 'amplifier_data' in result:
        for i, channel_info in enumerate(result['amplifier_channels']):
            # Check both custom_channel_name and native_channel_name
            if (channel_info.get('custom_channel_name') == channel_name or 
                channel_info.get('native_channel_name') == channel_name):
                if i < result['amplifier_data'].shape[0]:
                    return result['amplifier_data'][i, :]
    
    # If not found in amplifier data, check other signal groups
    signal_groups_data = [
        ('aux_input_channels', 'aux_input_data'),
        ('supply_voltage_channels', 'supply_voltage_data'),
        ('temp_sensor_channels', 'temp_sensor_data'),
        ('board_adc_channels', 'board_adc_data'),
        ('board_dig_in_channels', 'board_dig_in_data')
    ]
    
    for channel_group_key, data_key in signal_groups_data:
        if channel_group_key in result and data_key in result:
            for i, channel_info in enumerate(result[channel_group_key]):
                if (channel_info.get('custom_channel_name') == channel_name or 
                    channel_info.get('native_channel_name') == channel_name):
                    if i < result[data_key].shape[0]:
                        return result[data_key][i, :]
    
    print(f"Channel '{channel_name}' not found in any signal group")
    return None


def print_file_info(result):
    """
    Print basic information about the loaded RHD file.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from load_file()
    """
    print("\n=== RHD File Information ===")
    print(f"Sample rate: {int(result['frequency_parameters']['amplifier_sample_rate']/1000)} kHz")
    print(f"Number of amplifier channels: {len(result['amplifier_channels'])}")
    print(f"Each File Estimated Recording time: {result['t_amplifier'][-1]} seconds")
    
    # Print all channel names
    if 'amplifier_channels' in result:
        print(f"\nAmplifier channels ({len(result['amplifier_channels'])}):")
        for i, channel in enumerate(result['amplifier_channels']):
            print(f"  {i}: {channel['native_channel_name']}")


def load_experiment_data(experiment_dir):
    """
    Load all RHD data from an experiment directory.
    
    Parameters:
    -----------
    experiment_dir : str
        Path to experiment directory (e.g., 'data/intan/attempt1/d0_1')
        
    Returns:
    --------
    experiment_data : list
        List of loaded RHD files with their data
    """
    full_path = os.path.join(os.path.dirname(__file__), '..', experiment_dir)
    full_path = os.path.abspath(full_path)
    
    print(f"Loading experiment data from: {full_path}")
    return read_rhd_directory(full_path)


def rhd_folder_to_dataframe(folder_path, 
                            channel_names=None, 
                            resample_rate=None,
                            hk=False, 
                            hk_params={"env_threshold": 5,
                                       "kurtosis_threshold":3,
                                       "window_size":2.0,  # 1 second window
                                       "step_size":1,    # 0.5 second step
                                       "interpolate":True},
                            interpolate=False):
    """
    Convert all RHD files in a folder to a single pandas DataFrame.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing RHD files
    channel_names : list, optional
        List of specific channel names to include. If None, includes all amplifier channels
    resample_rate : float, optional
        If provided, resample the data to this rate (in Hz)
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with 'time' column and columns for each channel
    metadata : dict
        Dictionary containing metadata about the combined dataset
    """
    # Load all RHD files from the folder
    rhd_files_data = read_rhd_directory(folder_path)
    if not rhd_files_data:
        raise ValueError(f"No RHD files found in {folder_path}")
    
    # Initialize lists to store data
    all_data = []
    all_times = []
    file_info = []
    
    # Get channel information from the first file
    first_file = rhd_files_data[0]
    first_result = first_file[1]
    
    if 'amplifier_channels' not in first_result:
        raise ValueError("No amplifier channels found in the first RHD file")
    
    # Get available channel names
    available_channels = [ch['native_channel_name'] for ch in first_result['amplifier_channels']]
    
    # Use specified channels or all available channels
    if channel_names is None:
        selected_channels = available_channels
    else:
        # Check if requested channels exist
        missing_channels = [ch for ch in channel_names if ch not in available_channels]
        if missing_channels:
            print(f"Warning: Channels not found: {missing_channels}")
        selected_channels = [ch for ch in channel_names if ch in available_channels]
    
    if not selected_channels:
        raise ValueError("No valid channels found")
    
    print(f"Processing {len(rhd_files_data)} files with {len(selected_channels)} channels")
    print(f"Channels: {selected_channels}")
    
    cumulative_time = 0.0
    
    # Process each RHD file
    for filename, result, data_present in rhd_files_data:
        if not data_present:
            print(f"Skipping {filename} - no data present")
            continue
        try:
            # Get sample rate and recording duration
            sample_rate = get_sample_rate(result)
            
            # Get amplifier data
            if 'amplifier_data' not in result:
                print(f"Skipping {filename} - no amplifier data")
                continue
            
            amplifier_data = result['amplifier_data']
            num_samples = amplifier_data.shape[1] if amplifier_data.ndim > 1 else len(amplifier_data)
            
            # Create time vector for this file
            file_duration = num_samples / sample_rate
            time_vector = np.linspace(cumulative_time, cumulative_time + file_duration, num_samples, endpoint=False)
            
            # Extract data for selected channels
            file_data = {}
            for channel_name in selected_channels:
                # Find channel index
                channel_index = None
                for i, ch in enumerate(result['amplifier_channels']):
                    if ch['native_channel_name'] == channel_name:
                        channel_index = i
                        break
                
                if channel_index is not None and channel_index < amplifier_data.shape[0]:
                    file_data[channel_name] = amplifier_data[channel_index, :]
                else:
                    print(f"Warning: Channel {channel_name} not found in {filename}")
                    file_data[channel_name] = np.full(num_samples, np.nan)
            
            # Store data and metadata
            all_times.append(time_vector)
            all_data.append(file_data)
            
            file_info.append({
                'filename': filename,
                'sample_rate': sample_rate,
                'duration': file_duration,
                'num_samples': num_samples,
                'start_time': cumulative_time,
                'end_time': cumulative_time + file_duration
            })
            
            cumulative_time += file_duration
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found in any RHD files")
    
    # Concatenate all data
    print("Concatenating data from all files...")
    
    # Combine time vectors
    combined_time = np.concatenate(all_times)
    
    # Combine channel data
    combined_data = {'time': combined_time}
    
    for channel_name in selected_channels:
        channel_arrays = []
        for file_data in all_data:
            if channel_name in file_data:
                channel_arrays.append(file_data[channel_name])
            else:
                # Fill with NaN if channel is missing in this file
                expected_length = len(all_times[len(channel_arrays)])
                channel_arrays.append(np.full(expected_length, np.nan))
        
        combined_data[channel_name] = np.concatenate(channel_arrays)
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)

    if hk:
        # Apply Hilbert-Kurtosis artifact removal if requested
        print("Applying Hilbert-Kurtosis artifact removal...")
        from utils.signal_preprocessing import remove_artifacts_hilbert_kurtosis
        df = remove_artifacts_hilbert_kurtosis(
            df, 
            fs=sample_rate, 
            env_threshold=hk_params['env_threshold'], 
            kurtosis_threshold=hk_params['kurtosis_threshold'],
            window_size=1.0,  # 1 second window
            step_size=0.5,    # 0.5 second step
            interpolate=interpolate  # Interpolate immediately
        )
    
    # Resample if requested
    if resample_rate is not None and resample_rate != sample_rate:
        print(f"Resampling from {sample_rate} Hz to {resample_rate} Hz...")
        df = resample_dataframe(df, original_rate=sample_rate, target_rate=resample_rate)
    
    # Create metadata
    metadata = {
        'total_duration': cumulative_time,
        'total_samples': len(df),
        'channels': selected_channels,
        'files_processed': len(file_info),
        'file_info': file_info,
        'sample_rate': resample_rate if resample_rate else sample_rate,
        'original_sample_rate': sample_rate
    }
    
    print(f"Successfully created DataFrame with {len(df)} samples and {len(selected_channels)} channels")
    print(f"Total recording duration: {cumulative_time:.2f} seconds")
    
    return df, metadata


def resample_dataframe(df, original_rate, target_rate):
    """
    Resample a DataFrame to a different sampling rate.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time and channel data
    original_rate : float
        Original sampling rate in Hz
    target_rate : float
        Target sampling rate in Hz
        
    Returns:
    --------
    resampled_df : pandas.DataFrame
        Resampled DataFrame
    """
    from scipy import signal
    
    # Calculate resampling factor
    resample_factor = target_rate / original_rate
    new_length = int(len(df) * resample_factor)
    
    # Create new time vector
    new_time = np.linspace(df['time'].iloc[0], df['time'].iloc[-1], new_length)
    
    # Resample each channel
    resampled_data = {'time': new_time}
    
    for column in tqdm(df.columns):
        if column != 'time':
            # Use scipy's resample function
            resampled_channel = signal.resample(df[column].values, new_length)
            resampled_data[column] = resampled_channel
    
    return pd.DataFrame(resampled_data)


def dataframe_to_numpy(df, save_path=None, file_format='npz', pin_map=None, use_pin_names=True):
    """
    Convert DataFrame to numpy arrays and optionally save to file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time and channel data
    save_path : str, optional
        Path to save the numpy data (without extension)
    file_format : str, optional
        Format to save: 'npz' (compressed), 'npy' (single array), or 'csv'
    use_pin_names : bool, optional
        If True, convert channel names to pin names (e.g., 'B-008' -> 'pin_4')
        
    Returns:
    --------
    numpy_data : dict
        Dictionary containing numpy arrays for each column
    """
    print(f"Converting DataFrame to numpy arrays...")
    
    # Convert channel names to pin names if requested
    if use_pin_names:
        print("Converting channel names to pin names...")
        df_to_save = convert_channels_to_pins(df, pin_map)
    else:
        df_to_save = df
    
    # Convert DataFrame to dictionary of numpy arrays
    numpy_data = {}
    for column in df_to_save.columns:
        numpy_data[column] = df_to_save[column].values
        print(f"  - {column}: shape {numpy_data[column].shape}, dtype {numpy_data[column].dtype}")
    
    # Save to file if path is provided
    if save_path:
        if file_format.lower() == 'npz':
            # Save as compressed numpy archive (recommended)
            save_path_full = f"{save_path}.npz"
            np.savez_compressed(save_path_full, **numpy_data)
            print(f"✓ Saved as compressed numpy archive: {save_path_full}")
            
        elif file_format.lower() == 'npy':
            # Save each array separately
            for column, array in numpy_data.items():
                array_path = f"{save_path}_{column}.npy"
                np.save(array_path, array)
                print(f"✓ Saved {column}: {array_path}")
                
        elif file_format.lower() == 'csv':
            # Save as CSV (less efficient but human readable)
            csv_path = f"{save_path}.csv"
            df_to_save.to_csv(csv_path, index=False)
            print(f"✓ Saved as CSV: {csv_path}")
            
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'npz', 'npy', or 'csv'")
    
    return numpy_data


def load_numpy_data(file_path):
    """
    Load numpy data from saved file.
    
    Parameters:
    -----------
    file_path : str
        Path to the saved numpy file (.npz, .npy, or .csv)
        
    Returns:
    --------
    data : dict or pandas.DataFrame
        Loaded data
    """
    file_path = str(file_path)
    
    if file_path.endswith('.npz'):
        # Load compressed numpy archive
        loaded = np.load(file_path)
        data = {key: loaded[key] for key in loaded.files}
        # print(f"✓ Loaded numpy archive: {file_path}")
        # print(f"  Contains: {list(data.keys())}")
        return data
        
    elif file_path.endswith('.npy'):
        # Load single numpy array
        data = np.load(file_path)
        print(f"✓ Loaded numpy array: {file_path}, shape: {data.shape}")
        return data
        
    elif file_path.endswith('.csv'):
        # Load CSV as DataFrame
        data = pd.read_csv(file_path)
        print(f"✓ Loaded CSV: {file_path}, shape: {data.shape}")
        return data
        
    else:
        raise ValueError(f"Unsupported file format. Use .npz, .npy, or .csv")


def numpy_to_dataframe(numpy_data):
    """
    Convert numpy arrays back to DataFrame.
    
    Parameters:
    -----------
    numpy_data : dict
        Dictionary containing numpy arrays (from dataframe_to_numpy)
        
    Returns:
    --------
    df : pandas.DataFrame
        Reconstructed DataFrame
    """
    # Ensure 'time' column comes first if it exists
    if 'time' in numpy_data:
        columns_ordered = ['time'] + [col for col in numpy_data.keys() if col != 'time']
    else:
        columns_ordered = list(numpy_data.keys())
    
    # Create DataFrame with ordered columns
    df_data = {col: numpy_data[col] for col in columns_ordered}
    df = pd.DataFrame(df_data)
    
    # print(f"✓ Converted numpy data back to DataFrame: {df.shape}")
    return df

def rhd2dataframe(folder_path, pinmap):
    """
    Convert RHD files in a folder to a pandas DataFrame.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing RHD files
    pinmap : dict
        Mapping from pin numbers to Intan input names
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with 'time' column and columns for each channel
    """
    df, metadata = rhd_folder_to_dataframe(folder_path)
    
    # Convert channel names to pin names
    df = convert_channels_to_pins(df, pin_map=pinmap)
    
    return df, metadata


def save_experiment_numpy(folder_path, 
                          output_dir, 
                          channel_names=None, 
                          resample_rate=None, 
                          pin_map=None, 
                          use_pin_names=True, 
                          hk=True, 
                          hk_params={"env_threshold": 5,
                                       "kurtosis_threshold":3,
                                       "window_size":2,  # 1 second window
                                       "step_size":1,    # 0.5 second step
                                       "interpolate":True},
                          interpolate=True):
    """
    Process RHD folder and save as numpy arrays in one step.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing RHD files
    output_dir : str
        Directory to save the numpy files
    channel_names : list, optional
        List of specific channel names to include
    resample_rate : float, optional
        If provided, resample the data to this rate (in Hz)
    use_pin_names : bool, optional
        If True, convert channel names to pin names (e.g., 'B-008' -> 'pin_4')
        
    Returns:
    --------
    output_paths : dict
        Dictionary containing paths to saved files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert RHD to DataFrame
    print("Step 1: Converting RHD files to DataFrame...")
    df, metadata = rhd_folder_to_dataframe(folder_path, 
                                           channel_names=channel_names, 
                                           resample_rate=resample_rate, 
                                           hk=hk, 
                                           hk_params=hk_params,
                                           interpolate=interpolate)
    
    # Generate output file name based on folder
    folder_name = os.path.basename(os.path.abspath(folder_path))
    base_name = f"{folder_name}"
    
    # Save as numpy
    print("\nStep 2: Converting to numpy and saving...")
    numpy_path = os.path.join(output_dir, base_name)
    numpy_data = dataframe_to_numpy(df, save_path=numpy_path, file_format='npz', use_pin_names=use_pin_names)
    
    # Save metadata as JSON
    import json
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    
    # Convert numpy types to regular Python types for JSON serialization
    metadata_json = {}
    for key, value in metadata.items():
        if key == 'file_info':
            metadata_json[key] = value
        elif isinstance(value, (np.integer, np.floating)):
            metadata_json[key] = value.item()
        else:
            metadata_json[key] = value
    
    # Add pin mapping information to metadata
    if use_pin_names:
        metadata_json['pin_mapping'] = get_channel_mapping(pin_map)
        metadata_json['channel_to_pin_conversion'] = True
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    output_paths = {
        'numpy_data': f"{numpy_path}.npz",
        'metadata': metadata_path,
        'dataframe_shape': df.shape,
        'channels': list(df.columns),
        'total_duration': metadata['total_duration']
    }
    
    print(f"\n🎉 Successfully saved experiment data!")
    print(f"   - Numpy data: {output_paths['numpy_data']}")
    print(f"   - Metadata: {output_paths['metadata']}")
    print(f"   - Shape: {output_paths['dataframe_shape']}")
    print(f"   - Duration: {output_paths['total_duration']:.2f} seconds")
    if use_pin_names:
        print(f"   - Channel names converted to pin names")
    
    return output_paths


def load_experiment_numpy(numpy_file_path, metadata_file_path=None):
    """
    Load experiment data from saved numpy files.
    
    Parameters:
    -----------
    numpy_file_path : str
        Path to the .npz file containing the data
    metadata_file_path : str, optional
        Path to the JSON metadata file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded DataFrame
    metadata : dict
        Loaded metadata (if provided)
    """
    # Load numpy data
    numpy_data = load_numpy_data(numpy_file_path)
    
    # Convert back to DataFrame
    df = numpy_to_dataframe(numpy_data)
    
    # Load metadata if provided
    metadata = None
    if metadata_file_path:
        import json
        with open(metadata_file_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Loaded metadata: {metadata_file_path}")
    
    print(f"✓ Successfully loaded experiment data: {df.shape}")
    return df, metadata


def get_channel_mapping(pin_map=None):
    """
    Get the pin-to-channel mapping based on your device configuration.
    
    Returns:
    --------
    mapping : dict
        Dictionary mapping pin numbers to Intan input names
    """
    if pin_map is None:
        pin_map = {
            1: 'in11', 2: 'in10', 3: 'in9', 4: 'in8',
            5: 'in23', 6: 'in22', 7: 'in21', 8: 'in20',
            9: 'in12', 10: 'in13', 11: 'in14', 12: 'in15',
            13: 'in16', 14: 'in17', 15: 'in18', 16: 'in19'
        }
    
    return pin_map


def get_intan_to_pin_mapping(pin_map=None):
    """
    Get the reverse mapping from Intan input names to pin numbers.
    
    Returns:
    --------
    mapping : dict
        Dictionary mapping Intan input names to pin numbers
    """
    pin_to_intan = get_channel_mapping(pin_map)
    return {intan_name: pin for pin, intan_name in pin_to_intan.items()}


def channel_name_to_pin(channel_name, pin_map=None):
    """
    Convert a channel name (e.g., 'B-008', 'C-021') to pin number (e.g., 'pin_4', 'pin_7').
    
    Parameters:
    -----------
    channel_name : str
        Original channel name like 'B-008', 'C-021', etc.
        
    Returns:
    --------
    pin_name : str
        Pin name like 'pin_4', 'pin_7', or original name if no mapping found
    """
    # Extract the number part from channel name (e.g., '008' from 'B-008')
    import re
    match = re.search(r'(\d{3})$', channel_name)
    if not match:
        return channel_name  # Return original if no number found
    
    number_part = match.group(1)
    # Convert '008' to 'in8', '011' to 'in11', '021' to 'in21', etc.
    # Remove leading zeros from the number part
    intan_number = str(int(number_part))
    intan_name = f'in{intan_number}'
    
    # Get the mapping from intan names to pin numbers
    intan_to_pin = get_intan_to_pin_mapping(pin_map)
    
    if intan_name in intan_to_pin:
        pin_number = intan_to_pin[intan_name]
        return f'pin_{pin_number}'
    else:
        # If no mapping found, return original name
        print(f"Warning: No pin mapping found for {channel_name} ({intan_name})")
        return channel_name


def convert_channels_to_pins(df, pin_map=None):
    """
    Convert DataFrame column names from channel names to pin names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with channel names as columns
        
    Returns:
    --------
    df_pins : pandas.DataFrame
        DataFrame with pin names as columns
    """
    df_copy = df.copy()
    
    # Create mapping for column renaming
    column_mapping = {}
    for col in df.columns:
        if col == 'time':
            column_mapping[col] = col  # Keep time column as is
        else:
            pin_name = channel_name_to_pin(col, pin_map)
            column_mapping[col] = pin_name
            if pin_name != col:
                print(f"Mapped {col} -> {pin_name}")
    
    # Rename columns
    df_copy.rename(columns=column_mapping, inplace=True)
    return df_copy


# Example usage
if __name__ == "__main__":
    print("RHD Reader Module")
    print("Available functions:")
    print("- read_rhd_file(file_path)")
    print("- read_rhd_directory(directory_path)")
    print("- get_channel_data(result, channel_name)")
    print("- print_file_info(result)")
    print("- load_experiment_data(experiment_dir)")
    print("- rhd_folder_to_dataframe(folder_path, channel_names, resample_rate)")
    print("- dataframe_to_numpy(df, save_path, file_format, use_pin_names)")
    print("- load_numpy_data(file_path)")
    print("- save_experiment_numpy(folder_path, output_dir, use_pin_names)")
    print("- load_experiment_numpy(numpy_file_path, metadata_file_path)")
    print("- get_channel_mapping()")
    print("- channel_name_to_pin(channel_name)")
    print("- convert_channels_to_pins(df)")
    
    # Example usage for DataFrame creation
    print("\nExample usage:")
    print("# Basic DataFrame creation:")
    print("df, metadata = rhd_folder_to_dataframe('data/intan/attempt1/d0_1')")
    
    print("\n# Convert to numpy and save with pin names:")
    print("numpy_data = dataframe_to_numpy(df, 'output/my_data', 'npz', use_pin_names=True)")
    
    print("\n# One-step RHD to numpy conversion with pin names:")
    print("output_paths = save_experiment_numpy('data/intan/attempt1/d0_1', 'output', use_pin_names=True)")
    
    print("\n# Channel name to pin conversion:")
    print("pin_name = channel_name_to_pin('B-008')  # Returns 'pin_4'")
    
    print("\n# Load saved numpy data:")
    print("df_loaded, metadata = load_experiment_numpy('output/eeg_data_d0_1.npz', 'output/eeg_data_d0_1_metadata.json')")